#include "math.h"
#include "stdio.h"
#include <experimental/random>
#include <iostream>
#include <fstream>

__global__ void split(int N, int * splitIndex, int * pos, int * domain){
	int stride = blockDim.x;
	int index = threadIdx.x;
	
	__shared__ int splitSize;

	splitSize = 0;
	
	for (int sweep = 30; sweep > 0; sweep--){
		
		atomicExch(&splitSize, 0);
		
		__syncthreads();

		for (int i = index; i < N; i += stride){
			if (pos[i] > *splitIndex){
				domain[i] = 1;
				atomicAdd(&splitSize, 1);
			} else {
				domain[i] = 0;
			}
		}
		
		__syncthreads();
	
		// Only primary thread needs to execute this
		if (index == 0 && splitSize > N/2){
			//split = N
			*splitIndex -= 2 << sweep; 		
		}
		else if (index == 0){
			*splitIndex += 2 << sweep;
		}

		
		
	}	
}

int main() 
{	
	/// Parameters ///

	// 60k elements
	int N = 10<<16;
	// Inital split index guess
	int h_splitIndex = 0;


	/// Allocation /// 

	// Memory size
	int size = N * 3 * sizeof(int);
	
	// Host memory
	int* h_xPos = (int*)malloc(size);
	int* h_yPos = (int*)malloc(size);
	int* h_zPos = (int*)malloc(size);

	int* h_domain = (int*)malloc(size);

	// Device memory
	int* d_splitIndex;
	cudaMalloc(&d_splitIndex, sizeof(int));
	int* d_xPos;
	cudaMalloc(&d_xPos, size);
	int* d_domain;
	cudaMalloc(&d_domain, size);


	/// Initialisation ///

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
	cudaMemcpy(d_domain, h_domain, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_splitIndex, &h_splitIndex, sizeof(int), cudaMemcpyHostToDevice);

	// Do calculations
	split<<<1,256>>>(N, &h_splitIndex, h_xPos, h_domain);

	cudaDeviceSynchronize();

	// Copy to host
	cudaMemcpy(h_xPos, d_xPos, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_domain, d_domain, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(&h_splitIndex, d_splitIndex, sizeof(int), cudaMemcpyDeviceToHost);

	// Make sure all results
	//TODO: Is this the right place to call this?	
	//cudaDeviceSynchronize();
	
	// Free memory
	cudaFree(d_domain);
	cudaFree(d_xPos);
	cudaFree(d_splitIndex);


	/// Output ///

	printf("calculated one step");
	std::cout << h_splitIndex;
	remove( "out.dat" );
	std::ofstream Data("out.dat");
	
	for (int i = 0; i < N; i++){
		Data << h_xPos[3*i] << " " << h_yPos[3*i +1] << " " << h_domain[i]  << "\n";
	}

	Data.close();
    	return 0;
}

