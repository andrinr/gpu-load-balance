#include "Orb.h"

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

int Orb::build(blitz::Array<float, 2> &h_particles) 
{	
	h_particles = h_particles;

	unsigned int* h_domainID = (unsigned int*)malloc(unsignedSize);

	int size = h_particles->size();

	// Device memory
	int* d_particles;
	cudaMalloc(&d_particles, size);

	unsigned int* d_domainID;
	cudaMalloc(&d_domainID, unsignedSize);

	// Copy to device	
	cudaMemcpy(d_particles, h_particles, size, cudaMemcpyHostToDevice);
	
	// Random initial guess
	int splitPosition = 0;

	// TODO 

	// Questions: Directly use Blitz++ with CUDA?
	



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


