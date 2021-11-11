#include "math.h"
#include "stdio.h"
#include <experimental/random>
#include <iostream>
#include <fstream>
#include <assert.h>

__global__ void split(
	int nPositions, 
	int * positions, 
	int splitPosition, 
	unsigned int * splitSizes
){
	
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;

	// shared among theads in block
	extern __shared__ unsigned int s_splitSizes[];

	if (index < nPositions){
		int isLeft = positions[index] < splitPosition;
		s_splitSizes[threadIdx.x] = isLeft;
	}

	// sequential reduction, can be optimized further
        for (unsigned int s = blockDim.x/2; s > 0; s >>= 1){
                if (threadIdx.x < s){
			s_splitSizes[threadIdx.x] 
				+= s_splitSizes[threadIdx.x + s];
		}
		__syncthreads();
	}

	if (threadIdx.x == 0){
		splitSizes[blockIdx.x] = s_splitSizes[0];	
	}
}

// Sums all elements in array and stores result at index = 0
__global__ void sum(int size, unsigned int*values){
	for (unsigned int s = blockDim.x/2; s > 0; s >>= 1){
		if (threadIdx.x < s){
			values[threadIdx.x] += values[threadIdx.x + s];
		}
		__syncthreads();
	}	

}

__global__ void findDomainID(
	int nPositions, 
	int * positions, 
	int splitPosition, 
	unsigned int * domainIDs
){
	for (
		unsigned int i = threadIdx.x;
		i < nPositions; i += blockDim.x
	){
		int isLeft = positions[i] < splitPosition;
		domainIDs[i] = isLeft;
	}
}

int main() 
{	
	// 65k elements
	int p = 23;
	int N = 1<<p;

	// Memory size
	int size = N * sizeof(int);
	int unsignedSize = N * sizeof(unsigned int);

	// Host memory
	int* h_xPos = (int*)malloc(size);
	int* h_yPos = (int*)malloc(size);
	int* h_zPos = (int*)malloc(size);
	
	unsigned int* h_domainID = (unsigned int*)malloc(unsignedSize);

	// Device memory
	int* d_xPos;
	cudaMalloc(&d_xPos, size);

	int* d_yPos;
	cudaMalloc(&d_yPos, size);

	int* d_zPos;
	cudaMalloc(&d_zPos, size);

	unsigned int* d_domainID;
	cudaMalloc(&d_domainID, unsignedSize);

	// Data Initialisation
	for (int i = 0; i < N; i++){
		// xpos
		h_xPos[i] = std::experimental::randint(INT_MIN, INT_MAX);
		// ypos
		h_yPos[i] = std::experimental::randint(INT_MIN, INT_MAX);
		// zpos
		h_zPos[i] = std::experimental::randint(INT_MIN, INT_MAX);
	}

	std::cout << "initialization finished \n";

	// Copy to device	
	cudaMemcpy(d_xPos, h_xPos, size, cudaMemcpyHostToDevice);
	
	// Random initial guess
	int splitPosition = 0;

	unsigned int nThreads = 256;
	int nBlocks = (N + nThreads - 1) / nThreads;

	unsigned int h_splitSize;

	unsigned int* h_splitSizes = 
		(unsigned int*)malloc(nBlocks*sizeof(unsigned int));
	
	unsigned int* d_splitSizes;
	cudaMalloc(&d_splitSizes, nBlocks * sizeof(unsigned int));

	// Binary search for ideal splitPosition, assuming 32bit integers
	for (int i = 30; i >= 0; i--){
		
		// find split
		split<<<
			nBlocks, 
			nThreads, 
			nThreads*sizeof(unsigned int)
		>>>(
			N, 
			d_xPos, 
			splitPosition, 
			d_splitSizes
		);
		
		// copy back memory from device to host
		cudaMemcpy(
			h_splitSizes, 
			d_splitSizes, 
			nBlocks*sizeof(unsigned int), 
			cudaMemcpyDeviceToHost
		);
		
		// Sum up splitSizes from blocks
		h_splitSize = 0;
		for (unsigned int j = 0; j < nBlocks; j++){
			h_splitSize += h_splitSizes[j];	
		}

		// or also do it on GPU
		//sum<<<1, nThreads>>>(nBlocks, d_splitSizes);
		// Copy back to host
		//cudaMemcpy(&h_splitSize, d_splitSizes, sizeof(unsigned int), cudaMemcpyDeviceToHost);

		std::cout 
			<< splitPosition << " : " 
			<<  h_splitSize << "\n";	
		
		if (h_splitSize > 1<<(p-1)){
			splitPosition -= 1 << i;
		}
		else if(h_splitSize < 1<<(p-1)){
			splitPosition += 1 << i;
		}
		// ideal split has been found
		else{
			std::cout 
				<< "found split at position " 
				<< splitPosition << "\n";
			break;
		}
	}
 
	findDomainID
		<<<nBlocks, nThreads, nThreads*sizeof(unsigned int)>>>
 		(N, d_xPos, splitPosition, h_domainID);	
	
	cudaMemcpy(
		h_domainID, 
		d_domainID, 
		nBlocks*sizeof(unsigned int), 
		cudaMemcpyDeviceToHost
	);

	// Free memory
	cudaFree(d_xPos);
	cudaFree(d_splitSizes);

	// Output
	remove( "out.dat" );
	std::ofstream Data("out.dat");
	
	for (int i = 0; i < N; i++){
		Data 
			<< h_xPos[3*i] << " " 
			<< h_yPos[3*i +1] << " " 
			<< h_domainID[i]  << "\n";
	}

	Data.close();
    	return 0;
}


