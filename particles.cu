#include "math.h"
#include "stdio.h"
#include <iostream>
#include <fstream>

__global__ void step(int N, double *pos, double *vel, double h){
	int index = threadIdx.x;
	int stride = blockDim.x;
}	

__global__ void split(int N, double *xpos){
	// smt
}

int main() 
{
	int N = 10<<20; // 1M Elemets
	double *xpos;
	
	cudaMallocManaged(&xpos, N*sizeof(double));
	
	double pos [N*3];
	double vel [N*3];

	for (int i = 0; i < N; i++){
		pos[3*i] = std::rand();
		pos[3*i+1] = std::rand();
		pos[3*i+2] = std::rand();

		vel[3*i] = 0;
		vel[3*i+1] = 0;
		vel[3*i+2] = 0;
	}

	
	// :step<<<1,256>>>(N, pos, vel, 0.001); 

	cudaDeviceSynchronize();
	
	printf("calculated one step");

	cudaFree(pos);
	cudaFree(vel);
    	return 0;
}

