#include "condReduce.cuh"

template <unsigned int blockSize>
extern __device__ void warpReduce(volatile int *sdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize, bool leq>
extern __global__ void reduce(float *g_idata, uint *g_odata, float cut, int n) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize) + threadIdx.x;
    unsigned int gridSize = blockSize*gridDim.x;
    sdata[tid] = 0;

    while (i < n) {
        if (leq){
            sdata[tid] += (g_idata[i] <= cut);
        } else {
            sdata[tid] += (g_idata[i] > cut);
        }
        i += gridSize;
    }
    __syncthreads();

    if (blockSize >= 512) {
        if (tid < 256) {
            sdata[tid] += sdata[tid + 256];
        }
        __syncthreads();
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
    if (tid < 32) {
        warpReduce<blockSize>(sdata, tid);
    }
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

template <unsigned int blockSize, bool leq>
void conditionalReduce(
        float *g_idata,
        uint *g_odata,
        float cut,
        int n,
        int nBlocks,
        int nThreads,
        int sharedMemBytes,
        cudaStream_t stream) {

    reduce<blockSize, leq>
    <<<
    nBlocks,
    nThreads,
    sharedMemBytes,
    stream
    >>>
        (g_idata,
         g_odata,
         cut,
         n);
}
