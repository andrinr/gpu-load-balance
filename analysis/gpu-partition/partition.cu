#include <blitz/array.h>
#include "../../src/utils/condReduce.cuh"
#include "../../src/constants.h"
#include <thrust/partition.h>
// https://www.cse.chalmers.se/~tsigas/papers/GPU-Quicksort-jea.pdf
// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
// https://github.com/NVIDIA/cuda-samples/tree/master/Samples/2_Concepts_and_Techniques/scan
// https://onlinelibrary.wiley.com/doi/epdf/10.1002/cpe.3611


// 2 data elements per thread
// Code taken from: https://www.eecs.umich.edu/courses/eecs570/hw/parprefix.pdf
__device__ void scan(volatile unsigned int * s_idata, unsigned int thid, unsigned int n) {

    unsigned int offset = 1;
    for (unsigned int d = n>>1; d > 0; d >>= 1) // build sum in place up the tree
    {
        __syncthreads();
        if (thid < d)
        {
            unsigned int ai = offset*(2*thid+1)-1;
            unsigned int bi = offset*(2*thid+2)-1;
            s_idata[bi] += s_idata[ai];
        }
        offset *= 2;
    }
    if (thid == 0) { s_idata[n - 1] = 0; } // clear the last element
    for (unsigned int d = 1; d < n; d *= 2) // traverse down tree & build scan
    {
        offset >>= 1;
        __syncthreads();
        if (thid < d)
        {
            unsigned int ai = offset*(2*thid+1)-1;
            unsigned int bi = offset*(2*thid+2)-1;
            unsigned int t = s_idata[ai];
            s_idata[ai] = s_idata[bi];
            s_idata[bi] += t;
        }
    }

}

template <unsigned int blockSize>
__global__ void partition(
        unsigned int * g_offsetLessEquals,
        unsigned int * g_offsetGreater,
        float * g_idata,
        float * g_odata,
        float pivot,
        unsigned int nLeft) {
    __shared__ unsigned int s_lessEquals[blockSize * 2];
    __shared__ unsigned int s_greater[blockSize * 2];

    __shared__ unsigned int s_offsetLessEquals;
    __shared__ unsigned int s_offsetGreater;

    unsigned int tid = threadIdx.x;
    unsigned int n = blockSize * 2;
    unsigned int i = blockIdx.x * n + 2 * tid;
    unsigned int j = blockIdx.x * n + 2 * tid + 1;
    //unsigned int gridSize = blockSize*2*gridDim.x;

    bool f1 = g_idata[i] <= pivot;
    // potential to avoid bank conflicts here
    s_lessEquals[2*tid] = f1;
    s_greater[2*tid] = not f1;

    bool f2 = g_idata[j] <= pivot;
    s_lessEquals[2*tid+1] = f2;
    s_greater[2*tid+1] = not f2;

    __syncthreads();

    scan(s_lessEquals, tid, n);
    scan(s_greater, tid, n);

    __syncthreads();

    // Avoid another kernel
    if (tid == 0) {
        // result shared among kernel
        // atomicAdd returns old
        s_offsetLessEquals = atomicAdd(g_offsetLessEquals, s_lessEquals[n - 1]);
        s_offsetGreater = atomicAdd(g_offsetGreater, s_greater[n - 1]);
    }

    __syncthreads();

    // avoiding branch divergence
    unsigned int indexA = (s_lessEquals[2*tid] + s_offsetLessEquals) * f1 +
                          (s_greater[2*tid] + s_offsetGreater + nLeft) * (not f1);

    unsigned int indexB = (s_lessEquals[2*tid+1] + s_offsetLessEquals) * f2 +
                          (s_greater[2*tid+1] + s_offsetGreater + nLeft) * (not f2);

    g_odata[i] = (float) s_greater[2*tid];
    g_odata[j] = (float) s_greater[2*tid+1];


    //g_odata[indexA] = g_idata[i];
    //g_odata[indexB] = g_idata[j];

}

int main(int argc, char** argv) {

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int n = 1 << 9;
    unsigned int nLeft = 0;
    float cut = 0.5;

    float * h_idata = (float*)malloc(n * sizeof (float));
    float * h_odata = (float*)malloc(n * sizeof (float));

    for (int i = 0; i < n; i++) {
        h_idata[i] = (float)(rand())/(float)(RAND_MAX);
        nLeft += h_idata[i] < cut;
    }

    int nBlocks = (int) ceil((float) n / (N_THREADS * 2.0));
    printf("nBlocks: %d\n", nBlocks);

    float * d_idata;
    float * d_odata;

    printf("left %i \n", nLeft);
    CUDA_CHECK(cudaMalloc, (&d_idata, sizeof (float) * n));
    CUDA_CHECK(cudaMalloc, (&d_odata, sizeof (float) * n));
    cudaMemcpy(d_idata, h_idata, sizeof (float ) * n, cudaMemcpyHostToDevice);

    unsigned int * d_offsetLessEquals;
    unsigned int * d_offsetGreater;

    CUDA_CHECK(cudaMalloc,(&d_offsetLessEquals, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc,(&d_offsetGreater, sizeof(unsigned int)));

    CUDA_CHECK(cudaMemset,(d_offsetLessEquals, 0, sizeof (unsigned int)));
    CUDA_CHECK(cudaMemset,(d_offsetGreater, 0, sizeof (unsigned int)));

    cudaEventRecord(start);

    partition<N_THREADS><<<
            nBlocks,
            N_THREADS,
            N_THREADS * sizeof (unsigned int) * 8 + sizeof (unsigned int) * 2
            >>>(
            d_offsetLessEquals,
            d_offsetGreater,
            d_idata,
            d_odata,
            cut,
            nLeft);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << milliseconds << "\n";

    cudaMemcpy(h_odata, d_odata, sizeof (float ) * n, cudaMemcpyDeviceToHost);

    int sum = 0;

    /*for (int i = 0; i < n; ++i) {
        printf("%f ", h_idata[i]);
    }*/
    for (int i = 0; i < n; ++i) {
        printf("%f ", h_odata[i]);
        printf("%f ", h_idata[i]);
        /*if (h_odata[i] > cut) {
            throw std::runtime_error("Partition failed");
        }*/
    }

    printf("\n");

    cudaFree(d_idata);
    cudaFree(d_odata);
    cudaFree(d_offsetGreater);
    cudaFree(d_offsetLessEquals);

    free(h_idata);
    free(h_odata);

    cudaDeviceReset();
}