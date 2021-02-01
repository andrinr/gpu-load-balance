#include "math.h"
#include "stdio.h"
#include <experimental/random>
#include <iostream>
#include <fstream>

__global__ void split(int nPositions, int * positions, int splitPosition, int splitSize){
	int stride = blockDim.x;
	int index = threadIdx.x;
	
	//__shared__ int shared_splitSize;
	
	// Local splizSizes
        unsigned int l_splitSize;
	extern __shared__ unsigned int s_splitSizes;

        for (unsigned int i = index; i < N; i += stride){
                int isLeft = positions[i] > splitPosition;
                // Increment splitSize
                l_splitSize += isLeft;
        }

        s_splitSize[tid] = l_splitSize;

        for (unsigned int s = blockDim.x/2; s > 0; s >>= 1){
                s_splitSizes[tid] += s_splitSizes[tid + s];
	}
		
}

__global__ void findDomainID(int N, int * pos, int * splitPositions, int * domainIDs){
	
}

int main() 
{	
	/// Parameters ///

	// 60k elements
	int N = 10<<16;
	// Inital split index guess
	int h_splitPositions = 0;


	/// Allocation /// 

	// Memory size
	int size = N * sizeof(int);
	
	// Host memory
	int* h_xPos = (int*)malloc(size);
	int* h_yPos = (int*)malloc(size);
	int* h_zPos = (int*)malloc(size);

	int* h_domain = (int*)malloc(size);

	// Device memory
	int* d_xPos;
	cudaMalloc(&d_xPos, size);

	int* d_yPos;
	cudaMalloc(&d_yPos, size);

	int* d_zPos;
	cudaMalloc(&d_zPos, size);
	
	/// Data Initialisation ///
	for (int i = 0; i < N; i++){
		// xpos
		h_xPos[i] = std::experimental::randint(INT_MIN, INT_MAX);
		// ypos
		h_yPos[i] = std::experimental::randint(INT_MIN, INT_MAX);
		// zpos
		h_zPos[i] = std::experimental::randint(INT_MIN, INT_MAX);

		// domain
		h_domain[i] = 0;
	}

	// Copy to device	
	cudaMemcpy(d_xPos, h_xPos, size, cudaMemcpyHostToDevice);

	// Do calculations
	split<<<1,256>>>(N, h_xPos, h_splitPosition);
	
	// TODO: add second kernel to set domain
	
	// Copy to host
	// We do not need to copy back the xPositions
	//cudaMemcpy(h_xPos, d_xPos, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_domain, d_domain, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(&h_splitIndex, d_splitIndex, sizeof(int), cudaMemcpyDeviceToHost);

	// Make sure all results
	//TODO: Is this the right place to call this?
	// Not necessary here, since cudyMemcpy is already synchronized	
	//cudaDeviceSynchronize();
	
	// Free memory
	cudaFree(d_domain);
	cudaFree(d_xPos);
	cudaFree(d_splitIndex);


	/// Output ///
	std::cout << h_splitIndex;
	remove( "out.dat" );
	std::ofstream Data("out.dat");
	
	for (int i = 0; i < N; i++){
		Data << h_xPos[3*i] << " " << h_yPos[3*i +1] << " " << h_domain[i]  << "\n";
	}

	Data.close();
    	return 0;
}

