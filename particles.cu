#include "math.h"
#include "stdio.h"
#include <experimental/random>
#include <iostream>
#include <fstream>

__global__ void split(int N, int * pos){
	int stride = blockDim.x;
	int index = threadIdx.x;
}

int main() 
{
	int N = 10<<16; // 60k  Elemets
	int *xpos;
	int *domain;

	cudaMallocManaged(&xpos, N*sizeof(double));
	cudaMallocManaged(&domain, N*sizeof(int));
	
	int pos [N*3];
	int vel [N*3];

	for (int i = 0; i < N; i++){
		// xpos
		pos[3*i] = std::experimental::randint(INT_MIN, INT_MAX);
		xpos[i] = pos[3*i];
		domain[0] = 0;
		// ypos
		pos[3*i + 1] = std::experimental::randint(INT_MIN, INT_MAX);
		// zpos
		pos[3*i + 2] = std::experimental::randint(INT_MIN, INT_MAX);

		// xvel
		vel[3*i] = 0;
		// yvel
		vel[3*i+1] = 0;
		// zpos
		vel[3*i+2] = 0;
	}
	
	
	split<<<1,256>>>(N, xpos); 

	cudaDeviceSynchronize();
	
	printf("calculated one step");

	cudaFree(xpos);
	
	remove( "out.dat" );
	std::ofstream Data("out.dat");
	
	for (int i = 0; i < N; i++){
		Data << pos[3*i] << " " << pos[3*i +1] << " " << pos[3*i+2] << "\n";
	}

	Data.close();
    	return 0;
}

