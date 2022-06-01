//
// Created by andrin on 15/05/22.
//
#include "countService.h"
#include "serviceManager.h"
#inlucde "cell.h"

CountService::CountService() {};

template <unsigned int blockSize>
__device__ void warpReduce(volatile int *sdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

// todo: Rewrite this to work with cells
template <unsigned int blockSize>
__global__ void reduce(int *g_idata, int *g_odata, unsigned int n) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = 0;

    while (i < n) {
        sdata[tid] += g_idata[i] + g_idata[i+blockSize];
        i += gridSize;
    }

    __syncthreads();

    if (blockSize >= 512) {
        if (tid < 256) {
            sdata[tid] += sdata[tid + 256];
        } __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) {
            sdata[tid] += sdata[tid + 128];
        } __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) {
            sdata[tid] += sdata[tid + 64];
        } __syncthreads();
    }
    if (tid < 32)
        warpReduce(sdata, tid);
    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}

void CountService::run(void *inputBuffer, int inputBufferLength, void *outputBuffer, int outputBufferLength) {

    int n = 1000;

    Cell * h_cells;
    Cell * d_cells;
    cudaMalloc(&d_cells, sizeof (Cell) * n);
    cudaMemcpy(d_cells, h_cells, sizeof (Cell) * n, cudaMemcpyHostToDevice);

    float * h_particles;
    float * d_particles;
    cudaMalloc(&d_particles, sizeof (float) * n);
    cudaMemcpy(d_particles, h_particles, sizeof (float ) * n, cudaMemcpyHostToDevice);

    // Number of threads per block is limited

    // Need for cut service becomes clear here!



    cudaMalloc(&d_domainID, sizeof(Cell) * n);

}

std::tuple<int, int> CountService::getNBytes(int bufferLength) const {
    return std::make_tuple(bufferLength * sizeof(Cell), bufferLength * sizeof (int))
}

int CountService::getNOutputBytes(int outputBufferLength) const {
    return outputBufferLength * sizeof(int);
}

int CountService::getNInputBytes(int inputBufferLength) const {
    return inputBufferLength * sizeof(Cell)
}