#include "math.h"
#include "stdio.h"

__global__ void step(int N, double *pos, double *vel, double h){
	int index = threadIdx.x;
	int stride = blockDim.x;
	for (int i = index*3; i < N; i += stride * 3){
		vel[i*3] -= pos[i*3]*h;
		vel[i*3+1] -= pos[i*3]*h;
		vel[i*3+2] -= pos[i*3]*h;
		
		pos[i*3] += vel[i*3] * h;
		pos[i*3+1] += vel[i*3+1] * h;
		pos[i*3+2] += vel[i*3+2] * h;
	}
}

int main() 
{
	int N = 10<<20; // 1M Elemets
	double *pos, *vel;
	
	cudaMallocManaged(&pos, N*sizeof(double)*3);
	cudaMallocManaged(&vel, N*sizeof(double)*3);

	for (int i = 0; i < N; i++){
		pos[3*i] = std::rand();
		pos[3*i+1] = std::rand();
		pos[3*i+2] = std::rand();

		vel[3*i] = 0;
		vel[3*i+1] = 0;
		vel[3*i+2] = 0;
	}

    	step<<<1,256>>>(N, pos, vel, 0.001); 

	cudaDeviceSynchronize();
	
	printf("calculated one step");

	cudaFree(pos);
	cudaFree(vel);
    	return 0;
}

