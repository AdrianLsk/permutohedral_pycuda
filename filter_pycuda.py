import theano
from pycuda.compiler import SourceModule, compile
import pycuda.driver as cuda
from pycuda import gpuarray
import pycuda.autoinit
import numpy as np
import os
import sys
sys.path.append('/home/adrian/code/cutools/')
from gpustruct import GPUStruct
from jinja2 import Template
from time import time
import ctypes

# class PyCUDALatticeFiltereOp(theano.Op):
#
#     __props__ = ()
#
#     def make_node(self, inp):
#         inp = cuda.basic_ops.gpu_contiguous(
#            cuda.basic_ops.as_cuda_ndarray_variable(inp))
#         assert inp.dtype == "float32"
#         return theano.Apply(self, [inp], [inp.type()])
#
#     def make_thunk(self, node, storage_map, _, _2):
#         mod = SourceModule("""
#   #   __global__ void my_fct(float * i0, float * o0, int size) {
#   #   int i = blockIdx.x*blockDim.x + threadIdx.x;
#   #   if(i<size){
#   #       o0[i] = i0[i]*2;
#   #   }
#   # }""")
#   #       pycuda_fct = mod.get_function("my_fct")
#         inputs = [storage_map[v] for v in node.inputs]
#         outputs = [storage_map[v] for v in node.outputs]
#
#         # def thunk():
#         #     z = outputs[0]
#         #     if z[0] is None or z[0].shape != inputs[0][0].shape:
#         #         z[0] = cuda.CudaNdarray.zeros(inputs[0][0].shape)
#         #     grid = (int(np.ceil(inputs[0][0].size / 512.)), 1)
#         #     pycuda_fct(inputs[0][0], z[0], numpy.intc(inputs[0][0].size),
#         #                block=(512, 1, 1), grid=grid)
#         # return thunk
#
#         pd, vd, im, ref, w, h, accurate = inputs
#         z = outputs[0]

# filter(float *im, float *ref, int pd, int vd, int w, int h, bool accurate)
# filter(im, ref, refHeader.channels, imHeader.channels, imHeader.width, imHeader.height, accurate)
# i.e. pd=features dimension, vd=image num of channels
def filter_(pd, vd, w, h, im, values, accurate):
    # Initialize all parameters
    n = w*h
    blurVariance = 0 if accurate else 0.5

    # scaleFactor = (pd + 1) * np.sqrt(1./6 + blurVariance)
    scaleFactor = np.empty((pd,), dtype=np.float32)
    for i in range(pd):
        scaleFactor[i] = (pd+1)*np.sqrt((1.0/6 + blurVariance)/((i+1)*(i+2)))
    # scalef_gpu = cuda.mem_alloc(scaleFactor.nbytes)
    # cuda.memcpy_htod(scalef_gpu, scaleFactor) # scaleFactor.hostToDevice()
    scalef_gpu = gpuarray.to_gpu(scaleFactor)

    values = np.float32(im).ravel() # shape (n,vd) -> (n*vd,)
    # vals_gpu = cuda.mem_alloc(values.nbytes)
    # cuda.memcpy_htod(vals_gpu, values)
    vals_gpu = gpuarray.to_gpu(values)

    positions = np.float32(np.where(~np.isnan(values))).T.ravel() # shape (n,pd) -> (n*pd,)
    # pos_gpu = cuda.mem_alloc(positions.nbytes)
    # cuda.memcpy_htod(pos_gpu, positions)
    pos_gpu = gpuarray.to_gpu(positions)

    # matrixStruct = GPUStruct([(np.int32, 'index', 0),
    #                           (np.float32,'weight', 0.)])

    # allocate matrix structs on the gpu
    matrix_structs = map(lambda x: GPUStruct([(np.int32, 'index',0),
                                              (np.float32,'weight', 0.)]),
                         range(n*(pd+1)))
    map(lambda x: x.copy_to_gpu(), matrix_structs)

    # get pointer adresses of the structs
    struct_ptrs = np.asarray(map(lambda x: x.get_ptr(), matrix_structs),
                             dtype=np.intp)

    # allocate array for the matrix structs
    matrix_structs_gpu = gpuarray.to_gpu(struct_ptrs)

    # TODO need to sent the following instructions to the device
    # // Populate constant memory for hash helpers
    __host_two32 = np.ulonglong(1) << np.ulonglong(32) # unsigned long long int

    __host_div_c = [2*(n*(pd+1))]
    __host_div_c = np.uint32(__host_div_c)
    __host_div_l = [np.ceil(np.log(np.float32(__host_div_c) / np.log(2.0)))]
    __host_div_l = np.uint32(__host_div_l)

    __host_div_m = (__host_two32<<__host_div_l)/__host_div_c - __host_two32 + 1

    # __div_c = cuda.mem_alloc(__host_div_c.nbytes)
    # __div_l = cuda.mem_alloc(__host_div_l.nbytes)
    # __div_m = cuda.mem_alloc(__host_div_m.nbytes)
    #
    # cuda.memcpy_htod(__div_c, __host_div_c)
    # cuda.memcpy_htod(__div_l, __host_div_l)
    # cuda.memcpy_htod(__div_m, __host_div_m)

    # CUDA_SAFE_CALL(cudaMemcpyToSymbol((char*)&__div_c, &__host_div_c, sizeof(unsigned int)));
    # CUDA_SAFE_CALL(cudaMemcpyToSymbol((char*)&__div_l, &__host_div_l, sizeof(unsigned int)));
    # CUDA_SAFE_CALL(cudaMemcpyToSymbol((char*)&__div_m, &__host_div_m, sizeof(unsigned int)));

    # // Populate constant memory with hash of offset vectors
    hOffset_host = np.empty(pd+1, dtype=np.uint) # usigned int
    hOffset_host[:-1] = np.ones(pd, dtype=np.uint)
    offset = np.empty(pd+1, dtype=np.short) # signed short

    def hash(kd, key):
        k = 0
        for i in range(kd):
            k += key[i]
            k *= 2531011
        return k

    offset -= pd+1
    for i in range(pd+1):
        hOffset_host[i] = hash(pd, offset) # TODO get hash working
    offset += pd+1

    # hOffset = cuda.mem_alloc(hOffset_host.nbytes)
    # cuda.memcpy_htod(hOffset, hOffset_host)

    # CUDA_SAFE_CALL(cudaMemcpyToSymbol((char*)&hOffset, &hOffset_host, sizeof(unsigned int)*({{ pd }}+1)));

    cuda_dir = '/home/adrian/code/pydensecrf/densecrf/external/permutohedral_cuda'
    cuda_file = os.path.join(cuda_dir, 'permutohedral_pycuda.cu')
    with open(cuda_file) as f:
        f_txt = f.read()

    tpl = Template(f_txt)
    rendered_tpl = tpl.render(pd=pd, vd=vd)

    # cubin_file = compile(rendered_tpl, no_extern_c=True,
    #                      include_dirs=[cuda_dir])
    # print [txt for txt in cubin_file.split('\x00') if 'text' in txt]

    mod = SourceModule(rendered_tpl, no_extern_c=True, include_dirs=[cuda_dir])

    # createHashTable({{ pd }}, {{ vd }}+1, n*({{ pd }}+1));
    def createHashTable(kd, vd, capacity):
        table_capacity_gpu, _ = mod.get_global('table_capacity')
        cuda.memcpy_htod(table_capacity_gpu, np.uint([capacity]))

        # CUDA_SAFE_CALL(cudaMemcpyToSymbol(table_capacity,
        #           &capacity,
        #           sizeof(unsigned int)));

        table_vals_gpu, table_vals_size = mod.get_global('table_values') # pointer-2-pointer
        values_gpu = gpuarray.zeros((capacity*vd,1), dtype=np.float32)
        # values_gpu = gpuarray.zeros((capacity*vd,1), dtype=np.float32)
        # cuda.memset_d32(values_gpu.gpudata, 0, values_gpu.size)
        cuda.memcpy_dtod(table_vals_gpu, values_gpu.gpudata, table_vals_size)

        # float *values;
        # allocateCudaMemory((void**)&values, capacity*vd*sizeof(float));
        # CUDA_SAFE_CALL(cudaMemset((void *)values, 0, capacity*vd*sizeof(float)));
        # CUDA_SAFE_CALL(cudaMemcpyToSymbol(table_values,
        #                   &values,
        #                   sizeof(float *)));

        table_entries, table_entries_size = mod.get_global('table_entries')
        entries_gpu = gpuarray.empty((capacity*2,1), dtype=np.int)
        entries_gpu.fill(-1)
        # cuda.memset_d32(entries_gpu.gpudata, 1, entries_gpu.size)
        cuda.memcpy_dtod(table_entries, entries_gpu.gpudata, table_entries_size)

        # int *entries;
        # allocateCudaMemory((void **)&entries, capacity*2*sizeof(int));
        # CUDA_SAFE_CALL(cudaMemset((void *)entries, -1, capacity*2*sizeof(int)));
        # CUDA_SAFE_CALL(cudaMemcpyToSymbol(table_entries,
        #                   &entries,
        #                   sizeof(unsigned int *)));

        ########################################
        # Assuming LINEAR_D_MEMORY not defined #
        ########################################

        #  #ifdef LINEAR_D_MEMORY
        # char *ranks;
        # allocateCudaMemory((void**)&ranks, capacity*sizeof(char));
        # CUDA_SAFE_CALL(cudaMemcpyToSymbol(table_rank,
        #                   &ranks,
        #                   sizeof(char *)));
        #
        # signed short *zeros;
        # allocateCudaMemory((void**)&zeros, capacity*sizeof(signed short));
        # CUDA_SAFE_CALL(cudaMemcpyToSymbol(table_zeros,
        #                   &zeros,
        #                   sizeof(char *)));
        #
        # #else

        table_keys_gpu, table_keys_size = mod.get_global('table_keys')
        keys_gpu = gpuarray.zeros((capacity*kd,1), dtype=np.short)
        # keys_gpu = gpuarray.empty((capacity*kd,1), dtype=np.short)
        # cuda.memset_d32(keys_gpu.gpudata, 0, keys_gpu.size)
        cuda.memcpy_dtod(table_keys_gpu, keys_gpu.gpudata, table_keys_size)

        # signed short *keys;
        # allocateCudaMemory((void **)&keys, capacity*kd*sizeof(signed short));
        # CUDA_SAFE_CALL(cudaMemset((void *)keys, 0, capacity*kd*sizeof(signed short)));
        # CUDA_SAFE_CALL(cudaMemcpyToSymbol(table_keys,
        #                   &keys,

    createHashTable(pd, vd + 1, n * ( pd + 1))

    t = np.zeros(5)

    # get pointers of the variables on the device and assign them values
    __div_c, _ = mod.get_global('__div_c')
    cuda.memcpy_htod(__div_c, __host_div_c)
    __div_l, _ = mod.get_global('__div_l')
    cuda.memcpy_htod(__div_l, __host_div_l)
    __div_m, _ = mod.get_global('__div_m')
    cuda.memcpy_htod(__div_m, __host_div_m)

    hOffset, _ = mod.get_global('hOffset')
    cuda.memcpy_htod(hOffset, hOffset_host)

    #########################
    # create lattice matrix #
    #########################

    BLOCK_SIZE = (8, 8, 1)
    GRID_SIZE = ((w-1)/8+1, (h-1)/8+1, 1)

    create_mat_fn_name = "_Z12createMatrixiiPKfS0_S0_P11MatrixEntry"
    pycuda_create_mat_fn = mod.get_function(create_mat_fn_name)

    # createMatrix<<<blocks, blockSize>>>(w, h, positions.device,
    #                                     values.device,
    #                                     scaleFactor.device,
    #                                     matrix.device);

    t[0] = time()
    pycuda_create_mat_fn(np.int32(w), np.int32(h), pos_gpu, vals_gpu, scalef_gpu,
                         matrix_structs_gpu.gpudata,
                         block=BLOCK_SIZE, grid=GRID_SIZE)
                         # matrixStruct.get_ptr(), block=BLOCK_SIZE, grid=GRID_SIZE)
    t[0] = time() - t[0]

    ####################################
    # fix duplicate hash table entries #
    ####################################

    CLEAN_BLOCK_SIZE = (32, 1, 1)
    CLEAN_GRID_SIZE = ((n-1)/CLEAN_BLOCK_SIZE[0]+1, 2*(pd+1), 1)

    clean_hash_fn_name = "_Z14cleanHashTableiiP11MatrixEntry"
    pycuda_clean_hash_fn = mod.get_function(clean_hash_fn_name)

    # cleanHashTable<<<cleanBlocks, cleanBlockSize>>>({{ pd }}, 2*n*({{ pd }}+1),
    #                                                 matrix.device);
    # CUT_CHECK_ERROR("clean failed\n");

    t[1] = time()
    pycuda_clean_hash_fn(np.int32(pd), np.int32(2*n*(pd+1)),
                         matrix_structs_gpu.gpudata, # matrixStruct.get_ptr(),
                         block=CLEAN_BLOCK_SIZE, grid=CLEAN_GRID_SIZE)
    t[1] = time() - t[1]

    #########################
    # splat splits by color #
    #########################
    # ... need to extend the y coordinate to our blocks to represent that

    GRID_SIZE = (GRID_SIZE[0], GRID_SIZE[1] * ( pd + 1 )) # blocks.y *= pd+1;

    splat_cache_fn_name = "_Z10splatCacheiiPfP11MatrixEntry"
    pycuda_clean_hash_fn = mod.get_function(splat_cache_fn_name)

    # splatCache<<<blocks, blockSize>>>(w, h, values.device, matrix.device);
    # CUT_CHECK_ERROR("splat failed\n");

    t[2] = time()
    pycuda_clean_hash_fn(np.int32(w), np.int32(h), vals_gpu,
                         matrix_structs_gpu.gpudata, # matrixStruct.get_ptr(),
                         block=BLOCK_SIZE, grid=GRID_SIZE)
    t[2] = time() - t[2]


    if accurate:
        new_vals_size = n*(pd+1)*(vd+1)
        new_vals_gpu = gpuarray.zeros((new_vals_size,1), dtype=np.float32)
        # new_vals_gpu = gpuarray.empty((new_vals_size,1), dtype=np.float32)
        # cuda.memset_d32(new_vals_gpu.gpudata, 0, new_vals_gpu.size)

        # float *newValues;
        # allocateCudaMemory((void**)&(newValues), n*({{ pd }}+1)*({{ vd }}+1)*sizeof(float));
        # CUDA_SAFE_CALL(cudaMemset((void *)newValues, 0, n*({{ pd }}+1)*({{ vd }}+1)*sizeof(float)));

        ########
        # blur #
        ########

        blur_fn_name = "_Z4bluriPfiP11MatrixEntry"
        pycuda_blur_fn = mod.get_function(blur_fn_name)

        def swapHashTableValues(new_vals):
            table_vals, table_vals_size = mod.get_global('table_values') # (device_ptr, size_in_bytes)
            old_vals_gpu = cuda.mem_alloc(table_vals_size)
            # old_vals_gpu = gpuarray.empty((table_vals_size,1), )
            cuda.memcpy_dtod(old_vals_gpu, table_vals, table_vals_size)
            cuda.memcpy_dtod(table_vals, new_vals.gpudata, table_vals_size)
            return old_vals_gpu

        t[3] = time()
        for color in range(pd+1):
            pycuda_blur_fn(np.int32(n*(pd+1)), new_vals_gpu, np.int32(color),
                           matrix_structs_gpu.gpudata, # matrixStruct.get_ptr(),
                           block=CLEAN_BLOCK_SIZE, grid=CLEAN_GRID_SIZE)
            # TODO: newValues = swapHashTableValues(newValues);
            print color
            new_vals_gpu.gpudata = swapHashTableValues(new_vals_gpu)

            # blur<<<cleanBlocks, cleanBlockSize>>>(n*({{ pd }}+1), newValues, matrix.device, color);
            # CUT_CHECK_ERROR("blur failed\n");
        t[3] = time() - t[3]

    #########
    # slice #
    #########

    GRID_SIZE = (GRID_SIZE[0], GRID_SIZE[1] / ( pd + 1 ))

    slice_fn_name = "_Z4bluriPfP11MatrixEntryi"
    pycuda_slice_fn = mod.get_function(slice_fn_name)

    t[4] = time()
    pycuda_slice_fn(np.int32(w), np.int32(h), vals_gpu, matrix_structs_gpu.gpudata, # matrixStruct.get_ptr(),
                    block=BLOCK_SIZE, grid=GRID_SIZE)
    t[4] = time() - t[4]

    # slice<<<blocks, blockSize>>>(w, h, values.device, matrix.device);
    # CUT_CHECK_ERROR("slice failed\n");


    total_t = np.sum(t)
    print "Total time: {:3.3f} ms\n".format(total_t)
    # TODO: command (unsigned int) gpu_mem = GPU_MEMORY_ALLOCATION
    # print "Total GPU memory usage: %u bytes\n".format(gpu_mem)

    # cuda.memcpy_dtoh(values, vals_gpu) # values.deviceToHost();
    values = vals_gpu.get()

    def destroyHashTable():
        # assuming LINEAR_D_MEMORY not defined
        table_keys, _ = mod.get_global('table_keys')
        table_keys.free()

        table_vals, _ = mod.get_global('table_values')
        table_vals.free()

        table_ents, _ = mod.get_global('table_entries')
        table_ents.free()
    destroyHashTable() # TODO: command destroyHashTable();

    # Python deinitialises objects as soon as the reference count for them
    # becomes zero. If you need to do it explicitly, I think just "del
    # gpuarray_obj" will be enough.

    return values