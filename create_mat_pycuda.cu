
#define BLOCK_SIZE 64

#define _DEBUG
#include "cutil.h"
#include <cuda_runtime.h>
#include <stdio.h>

#include "cuda_memory.h"
//#ifdef WIN32
//#include "win32time.h"
//#else
//#include <sys/time.h>
//#endif

#include "MirroredArray.h"
#include "hash_table.cu"
  
//#ifdef LIBRARY
//extern "C"

struct MatrixEntry {
    int index;
    float weight;
};

//template<int {{ pd }}>
// Permutohedral::init
__global__ void createMatrix(const int w, const int h,
                             const float *positions,
                             const float *values,
                             const float *scaleFactor,
                             MatrixEntry *matrix) {
    // scanline order
    //const int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    // 8x8 blocks    
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int threadId = threadIdx.y*blockDim.x + threadIdx.x;
    const int idx = y*w + x;
    const bool outOfBounds = (x >= w) || (y >= h);

    float myElevated[{{ pd }}+1];
    const float *myPosition = positions + idx*{{ pd }};

    int myGreedy[{{ pd }}+1];
    int myRank[{{ pd }}+1];

    float myBarycentric[{{ pd }}+2];
    __shared__ short keys[{{ pd }}*BLOCK_SIZE];
    short *myKey = keys + threadId * {{ pd }};

    if (!outOfBounds) {

	myElevated[{{ pd }}] = -{{ pd }}*(myPosition[{{ pd }}-1])*scaleFactor[{{ pd }}-1];
	for (int i = {{ pd }}-1; i > 0; i--) {
	    myElevated[i] = (myElevated[i+1] - 
			     i*(myPosition[i-1])*scaleFactor[i-1] + 
			     (i+2)*(myPosition[i])*scaleFactor[i]);
	}
	myElevated[0] = myElevated[1] + 2*(myPosition[0])*scaleFactor[0];
	
		
	// find the closest zero-colored lattice point

	// greedily search for the closest zero-colored lattice point
	signed short sum = 0;
	for (int i = 0; i <= {{ pd }}; i++) {
	    float v = myElevated[i]*(1.0f/({{ pd }}+1));
	    float up = ceilf(v) * ({{ pd }}+1);
	    float down = floorf(v) * ({{ pd }}+1);
	    if (up - myElevated[i] < myElevated[i] - down) {
		myGreedy[i] = (signed short)up;
	    } else {
		myGreedy[i] = (signed short)down;
	    }
	    sum += myGreedy[i];
	}
	sum /= {{ pd }}+1;
	
	// sort differential to find the permutation between this simplex and the canonical one
	for (int i = 0; i <= {{ pd }}; i++) {
	    myRank[i] = 0;
	    for (int j = 0; j <= {{ pd }}; j++) {
		if (myElevated[i] - myGreedy[i] < myElevated[j] - myGreedy[j] ||
		    (myElevated[i] - myGreedy[i] == myElevated[j] - myGreedy[j]
		     && i > j)) {
		    myRank[i]++;
		}
	    }
	}
	
	if (sum > 0) { // sum too large, need to bring down the ones with the smallest differential
	    for (int i = 0; i <= {{ pd }}; i++) {
		if (myRank[i] >= {{ pd }} + 1 - sum) {
		    myGreedy[i] -= {{ pd }}+1;
		    myRank[i] += sum - ({{ pd }}+1);
		} else {
		    myRank[i] += sum;
		}
	    }
	} else if (sum < 0) { // sum too small, need to bring up the ones with largest differential
	    for (int i = 0; i <= {{ pd }}; i++) {
		if (myRank[i] < -sum) {
		    myGreedy[i] += {{ pd }}+1;
		    myRank[i] += ({{ pd }}+1) + sum;
		} else {
		    myRank[i] += sum;
		}
	    }
	}

        #ifdef LINEAR_D_MEMORY
	for (int i = 0; i <= {{ pd }}; i++) {
	    table_zeros[idx*({{ pd }}+1)+i] = myGreedy[i];
	    table_rank[idx*({{ pd }}+1)+i] = myRank[i];
	}
	#endif

	// turn delta into barycentric coords
	for (int i = 0; i <= {{ pd }}+1; i++) {
	    myBarycentric[i] = 0;
	}
	
	for (int i = 0; i <= {{ pd }}; i++) {
	    float delta = (myElevated[i] - myGreedy[i]) * (1.0f/({{ pd }}+1));
	    myBarycentric[{{ pd }}-myRank[i]] += delta;
	    myBarycentric[{{ pd }}+1-myRank[i]] -= delta;
	}
	myBarycentric[0] += 1.0f + myBarycentric[{{ pd }}+1];
    }

    #ifdef USE_ADDITIVE_HASH
    unsigned int cumulative_hash = hash<{{ pd }}>(myGreedy);
    #endif
    for (int color = 0; color <= {{ pd }}; color++) {
	// Compute the location of the lattice point explicitly (all but
	// the last coordinate - it's redundant because they sum to zero)
	if (!outOfBounds) {
	    for (int i = 0; i < {{ pd }}; i++) {
		myKey[i] = myGreedy[i] + color;
		if (myRank[i] > {{ pd }}-color) myKey[i] -= ({{ pd }}+1);
	    }
	}

	#ifdef USE_ADDITIVE_HASH
	for (int i = 0; i < {{ pd }}; i++) {
	    if (myRank[i] == {{ pd }}-color) cumulative_hash += hOffset[i];
	}
	#endif
	
	if (!outOfBounds) {
	    MatrixEntry r;
	    #ifdef USE_ADDITIVE_HASH
	    r.index = hashTableInsert<{{ pd }}>(cumulative_hash, myKey, idx*({{ pd }}+1)+color);
	    #else
	    r.index = hashTableInsert<{{ pd }}>(myKey, idx*({{ pd }}+1)+color);
	    #endif
	    r.weight = myBarycentric[color];
	    matrix[idx*({{ pd }}+1) + color] = r;
	}
    }    
}
//#endif
