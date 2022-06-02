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
__global__ void reduce(int *g_idata, int *g_odata, float cut) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
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

int main(int argc, char** argv) {

    int n = 1 << 20;

    std::cout << n << "\n";

    blitz::Array<float, 2> p = blitz::Array<float, 2>(n, 3);
    srand(seed);
    for (int i = 0; i < p.rows(); i++) {
        for (int d = 0; d < 3; d++) {
            p(i,d) = (float)(rand())/(float)(RAND_MAX);
        }
    }

    int nBlocks;
    int nThreads = 512;

    float * h_particles = p.data();
    float * d_particles;
    int * d_sums;
    int * h_sums;
    cudaMalloc(&d_particles, sizeof (float) * n);
    cudaMemcpy(d_particles, h_particles, sizeof (float ) * n, cudaMemcpyHostToDevice);

    cudaMalloc(&d_sums, sizeof (int) * n);

    // Number of threads per block is limited

    // Need for cut service becomes clear here!

    reduce<512><<<n/ nThreads, nThreads>>>(d_particles, d_sums);

    cudaMemcpy(h_sums, d_sums, sizeof (int ) * n, cudaMemcpyDeviceToHost);

}
