
#include "stdio.h"

__global__ void step(N, pos, vel, h){
	for (int i = 0; i < N; i++){
		vel[i*3] += std::rand()*10<<-5 * h;
		vel[i*3+1] += std::rand()*10<<-5 * h;
		vel[i*3+2] += std::rand()*10<<-5 * h;

		
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

    	step<<<1,1>>>(N, pos, vel, 10<<-3); 

	cudaFree(x);
	cudaFree(y);
	cudaFree(z);
    	return 0;
}
