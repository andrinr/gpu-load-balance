#include <blitz/array.h>
#include "../../src/utils/condReduce.cuh"
#include "../../src/constants.h"
#include <thrust/partition.h>
// https://www.cse.chalmers.se/~tsigas/papers/GPU-Quicksort-jea.pdf
// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
// https://github.com/NVIDIA/cuda-samples/tree/master/Samples/2_Concepts_and_Techniques/scan
// https://onlinelibrary.wiley.com/doi/epdf/10.1002/cpe.3611

#define N_THREADS 256
#define SHARED_MEMORY_BANKS 8
#define LOG_MEM_BANKS 3
#define CONFLICT_FREE_OFFSET(n) ((n) >> SHARED_MEMORY_BANKS + (n) >> (2 * LOG_MEM_BANKS))
// 2 data elements per thread
template <unsigned int blockSize>
__device__ void scan(volatile unsigned int * s_idata, unsigned int thid) {

    int ai = thid;
    int bi = thid + (blockSize);
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
    temp[ai + bankOffsetA] = g_idata[ai];
    temp[bi + bankOffsetB] = g_idata[bi];

    int offset = 1;
    for (int d = blockSize>>1; d > 0; d >>= 1) { // build sum in place up the tree
        __syncthreads();
        if (thid < d) {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            s_idata[bi] += s_idata[ai];
        }
        offset *= 2;
    }
    if (thid==0) {
        s_idata[blockSize - 1 + CONFLICT_FREE_OFFSET(blockSize * 2 - 1)] = 0;
    }

    for (int d = 1; d < blockSize; d *= 2) {// traverse down tree & build scan
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            float t = s_idata[ai];
            s_idata[ai] = s_idata[bi];
            s_idata[bi] += t;
        }
    }  __syncthreads();
}

template <unsigned int blockSize>
__global__ void partition(
        unsigned int * g_totalLeft,
        unsigned int * g_totalRight,
        float * g_idata,
        float * g_odata,
        float pivot,
        unsigned int nLeft) {
    extern __shared__ unsigned int s_lqPivot[];
    extern __shared__ unsigned int s_gPivot[];
    extern __shared__ float s_res[];

    extern __shared__ unsigned int offsetLeq;
    extern __shared__ unsigned int offsetG;

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize * 2)+threadIdx.x;
    //unsigned int gridSize = blockSize*2*gridDim.x;

    unsigned int f1 = g_idata[2*i] < pivot;
    s_lqPivot[2 * tid] = f1;
    s_gPivot[2 * tid] = 1-f1;

    unsigned int f2 = g_idata[2 * i + 1] < pivot;
    s_lqPivot[2 * tid + 1] = f2;
    s_gPivot[2 * tid + 1] = 1-f2;

    __syncthreads();

    scan<blockSize>(s_lqPivot, tid);
    scan<blockSize>(s_gPivot, tid);

    __syncthreads();

    // Avoid another kernel
    if (tid == 0) {
        offsetLeq = atomicAdd(g_totalLeft, s_lqPivot[blockSize * 2 - 1]);
        offsetG = atomicAdd(g_totalRight, s_gPivot[blockSize * 2 - 1]);
    }

    __syncthreads();

    // avoiding branch divergence
    g_odata[
            (s_lqPivot[2*tid] + offsetLeq) * f1 +
            (s_gPivot[2*tid] + offsetG + nLeft) * (1-f1)] = g_idata[2*i];

    g_odata[
            (s_lqPivot[2*tid+1] + offsetLeq) * f2 +
            (s_gPivot[2*tid+1] + offsetG + nLeft) * (1-f2)] = g_idata[2*i+1];

}


int main(int argc, char** argv) {

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int n = 1 << 13;
    unsigned int nLeft = 0;
    float cut = 0.5;

    float * pos = (float*)malloc(n);
    for (int i = 0; i < n; i++) {
        pos[i] = (float)(rand())/(float)(RAND_MAX);
        nLeft += pos[i] < cut;
    }

    int nBlocks = (int) ceil((float) n / (N_THREADS * 2.0));

    float * d_idata;
    float * d_odata;
    CUDA_CHECK(cudaMalloc, (&d_idata, sizeof (float) * n));
    CUDA_CHECK(cudaMalloc, (&d_odata, sizeof (float) * n));
    cudaMemcpy(d_idata, pos, sizeof (float ) * n, cudaMemcpyHostToDevice);

    unsigned int * d_totalLeq;
    unsigned int * d_totalG;

    cudaMalloc(&d_totalLeq, sizeof(uint));
    cudaMalloc(&d_totalG, sizeof(uint));

    cudaMemset(&d_totalLeq, 0, sizeof(uint));
    cudaMemset(&d_totalG, 0, sizeof(uint));

    cudaEventRecord(start);

    partition<N_THREADS><<<
            nBlocks,
            N_THREADS,
            N_THREADS * sizeof (uint) * 2 >>>(
            d_totalLeq,
            d_totalG,
            d_idata,
            d_odata,
            cut,
            nLeft);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << milliseconds << "\n";

    cudaMemcpy(pos, d_odata, sizeof (float ) * n, cudaMemcpyDeviceToHost);

    int sum = 0;

    for (int i = 0; i < nLeft; ++i) {
        if (pos[i] > cut) {
            throw std::runtime_error("Error");
        }
    }

    std::cout << "is " << sum << " should be " << n / 2 << " \n";

    cudaFree(d_idata);
    cudaFree(d_odata);

    free(pos);

    cudaDeviceReset();
}