#include "math.h"
#include "stdio.h"
#include <experimental/random>
#include <iostream>
#include <fstream>
#include <assert.h>

__global__ void split(int nPositions, int * positions, int splitPosition, unsigned int * splitSize){
	
	unsigned int tid = threadIdx.x;
	
        unsigned int l_splitSize = 0;
	extern __shared__ unsigned int s_splitSizes[];

        for (unsigned int i = threadIdx.x; i < nPositions; i += blockDim.x){
                int isLeft = positions[i] > splitPosition;
                // Increment splitSize
                l_splitSize += isLeft;
        }

        s_splitSizes[tid] = l_splitSize;

        for (unsigned int s = blockDim.x/2; s > 0; s >>= 1){
                if (tid < s){
			s_splitSizes[tid] += s_splitSizes[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0){
		*splitSize = s_splitSizes[0];	
	}

	assert(*splitSize > 0);
}

__global__ void findDomainID(int nPositions, int * positions, int splitPosition, unsigned int * domainIDs){
	for (unsigned int i = thread.x; i < nPositions; i += blockDim.x){
		int isLeft = positions[i] > splitPosition;
		domainIDs[i] = isLeft;
	}
}

int main() 
{	
	// 60k elements
	int N = 10<<16;

	// Memory size
	int size = N * sizeof(int);
	
	// Host memory
	int* h_xPos = (int*)malloc(size);
	int* h_yPos = (int*)malloc(size);
	int* h_zPos = (int*)malloc(size);

	// Device memory
	int* d_xPos;
	cudaMalloc(&d_xPos, size);

	int* d_yPos;
	cudaMalloc(&d_yPos, size);

	int* d_zPos;
	cudaMalloc(&d_zPos, size);

	unsigned int* d_splitSize;
	cudaMalloc(&d_splitSize, sizeof(unsigned int));

	// Data Initialisation
	for (int i = 0; i < N; i++){
		// xpos
		h_xPos[i] = std::experimental::randint(INT_MIN, INT_MAX);
		// ypos
		h_yPos[i] = std::experimental::randint(INT_MIN, INT_MAX);
		// zpos
		h_zPos[i] = std::experimental::randint(INT_MIN, INT_MAX);
	}

	// Copy to device	
	cudaMemcpy(d_xPos, h_xPos, size, cudaMemcpyHostToDevice);
	
	// Do calculations
	int h_splitPosition = 0;

	split<<<1,256>>>(N, d_xPos, h_splitPosition, d_splitSize);
	
	// Copy back to host	
	unsigned int h_splitSize = 0;
	cudaMemcpy(&h_splitSize, d_splitSize, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	
	// Free memory
	cudaFree(d_xPos);
	cudaFree(d_splitSize);

	// Output
	std::cout << h_splitSize;
	remove( "out.dat" );
	//std::ofstream Data("out.dat");
	
	//for (int i = 0; i < N; i++){
	//	Data << h_xPos[3*i] << " " << h_yPos[3*i +1] << " " << h_domain[i]  << "\n";
	//}

	//Data.close();
    	return 0;
}

