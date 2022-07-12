#include <blitz/array.h>
#include "../../src/utils/condReduce.cuh"
#include "../../src/constants.h"
#include <thrust/partition.h>
// https://www.cse.chalmers.se/~tsigas/papers/GPU-Quicksort-jea.pdf
// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
// https://github.com/NVIDIA/cuda-samples/tree/master/Samples/2_Concepts_and_Techniques/scan
// https://onlinelibrary.wiley.com/doi/epdf/10.1002/cpe.3611

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
        unsigned int * g_permutations,
        float pivot,
        unsigned int nLeft,
        unsigned int n) {
    __shared__ unsigned int s_lessEquals[blockSize * 2];
    __shared__ unsigned int s_greater[blockSize * 2];

    __shared__ unsigned int s_offsetLessEquals;
    __shared__ unsigned int s_offsetGreater;

    unsigned int tid = threadIdx.x;

    unsigned int i = blockIdx.x * blockSize * 2 + 2 * tid;
    unsigned int j = blockIdx.x * blockSize * 2 + 2 * tid + 1;
    //unsigned int gridSize = blockSize*2*gridDim.x;

    bool f1, f2;
    if (i < n) {
        f1 = g_idata[i] <= pivot;
        f2 = not f1;
        // potential to avoid bank conflicts here
        s_lessEquals[2*tid] = f1;
        s_greater[2*tid] = f2;
    }
    else {
        f1 = false;
        f2 = false;
        s_lessEquals[2*tid] = 0;
        s_greater[2*tid] = 0;
    }

    bool f3, f4;
    if (j < n) {
        f3 = g_idata[j] <= pivot;
        f4 = not f3;
        // potential to avoid bank conflicts here
        s_lessEquals[2*tid+1] = f3;
        s_greater[2*tid+1] = f4;
    }
    else {
        f3 = false;
        f4 = false;
        s_lessEquals[2*tid+1] = 0;
        s_greater[2*tid+1] = 0;
    }

    __syncthreads();

    scan(s_lessEquals, tid, blockSize * 2 );
    scan(s_greater, tid, blockSize * 2);

    __syncthreads();

    // Avoid another kernel
    if (tid == blockSize - 1) {
        // result shared among kernel
        // atomicAdd returns old
        // exclusive scan does not include the last element
        s_offsetLessEquals = atomicAdd(g_offsetLessEquals, s_lessEquals[blockSize * 2 - 1] + f3);
        s_offsetGreater = atomicAdd(g_offsetGreater, s_greater[blockSize * 2 - 1] + f4);
    }

    __syncthreads();

    // avoiding warp divergence
    unsigned int indexA = (s_lessEquals[2*tid] + s_offsetLessEquals) * f1 +
                          (s_greater[2*tid] + s_offsetGreater + nLeft) * f2;

    unsigned int indexB = (s_lessEquals[2*tid+1] + s_offsetLessEquals) * f3 +
                          (s_greater[2*tid+1] + s_offsetGreater + nLeft) * f4;

    if (i < n) {
        g_odata[indexA] = g_idata[i];
        g_permutations[i] = indexA;
    }

    if (j < n) {
        g_odata[indexB] = g_idata[j];
        g_permutations[j] = indexB;
    }
}


template <unsigned int blockSize>
__global__ void permute(
        float * g_idata,
        float * g_odata,
        unsigned int * g_permutations,
        int n) {
    unsigned int tid = threadIdx.x;

    unsigned int i = blockIdx.x * blockSize * 2 + 2 * tid;
    unsigned int j = blockIdx.x * blockSize * 2 + 2 * tid + 1;
    //unsigned int gridSize = blockSize*2*gridDim.x;

    if (i < n) {
        g_odata[g_permutations[i]] = g_idata[i];
    }

    if (j < n) {
        g_odata[g_permutations[j]] = g_idata[j];
    }
}


int main(int argc, char** argv) {

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    unsigned int n = (1 << 10) - 100;
    unsigned int nLeft = 0;
    float cut = 0.5;

    float * h_dataX = (float*)malloc(n * sizeof (float));
    float * h_dataY = (float*)malloc(n * sizeof (float));
    float * h_dataZ = (float*)malloc(n * sizeof (float));
    unsigned int * h_permutations = (unsigned int *)malloc(n * sizeof (unsigned int));

    srand(0);

    for (int i = 0; i < n; i++) {
        h_dataX[i] = (float)(rand())/(float)(RAND_MAX);
        h_dataY[i] = (float)(rand())/(float)(RAND_MAX);
        h_dataZ[i] = (float)(rand())/(float)(RAND_MAX);
        nLeft += h_dataX[i] < cut;

        //printf("%f\n", h_idata[i]);
    }

    for (int i = 0; i < n; i++) {
        if (h_dataX[i] <0.5 ) {
            h_dataY[i] = 4.0;
            h_dataZ[i] = 3.0;
        }
        else {
            h_dataY[i] = 10.0;
            h_dataZ[i] = 11.0;
        }
    }

    int nBlocks = (int) ceil((float) n / (N_THREADS * 2.0));

    float * d_dataX;
    float * d_dataY;
    float * d_dataZ;
    float * d_dataT;
    unsigned int * d_permutations;

    CUDA_CHECK(cudaMalloc, (&d_dataX, sizeof (float) * n));
    CUDA_CHECK(cudaMalloc, (&d_dataY, sizeof (float) * n));
    CUDA_CHECK(cudaMalloc, (&d_dataZ, sizeof (float) * n));
    CUDA_CHECK(cudaMalloc, (&d_dataT, sizeof (float) * n));
    CUDA_CHECK(cudaMalloc, (&d_permutations, sizeof (unsigned int) * n));
    cudaMemcpy(d_dataX, h_dataX, sizeof (float ) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataY, h_dataY, sizeof (float ) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataZ, h_dataZ, sizeof (float ) * n, cudaMemcpyHostToDevice);

    unsigned int * d_offsetLessEquals;
    unsigned int * d_offsetGreater;

    CUDA_CHECK(cudaMalloc,(&d_offsetLessEquals, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc,(&d_offsetGreater, sizeof(unsigned int)));

    CUDA_CHECK(cudaMemset,(d_offsetLessEquals, 0, sizeof (unsigned int)));
    CUDA_CHECK(cudaMemset,(d_offsetGreater, 0, sizeof (unsigned int)));
    //CUDA_CHECK(cudaMemset,(d_dataT, 255, sizeof (float ) * n));
    //CUDA_CHECK(cudaMemset,(d_dataY, 255, sizeof (float ) * n));

    cudaEventRecord(start);

    partition<N_THREADS><<<
            nBlocks,
            N_THREADS,
            (N_THREADS) * sizeof (unsigned int) * 4 + sizeof (unsigned int) * 2
            >>>(
            d_offsetLessEquals,
            d_offsetGreater,
            d_dataX,
            d_dataT,
            d_permutations,
            cut,
            nLeft,
            n);

    cudaMemcpy(d_dataX, d_dataT, sizeof (float ) * n, cudaMemcpyHostToHost);

    permute<N_THREADS><<<
        nBlocks,
        N_THREADS
    >>>(
            d_dataY,
            d_dataT,
            d_permutations,
            n
    );

    cudaMemcpy(d_dataY, d_dataT, sizeof (float ) * n, cudaMemcpyHostToHost);

    permute<N_THREADS><<<
    nBlocks,
    N_THREADS
    >>>(
            d_dataZ,
            d_dataT,
            d_permutations,
            n
    );

    cudaMemcpy(d_dataZ, d_dataT, sizeof (float ) * n, cudaMemcpyHostToHost);


    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_dataY, d_dataY, sizeof (float ) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dataZ, d_dataZ, sizeof (float ) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_permutations, d_permutations, sizeof (unsigned int) * n, cudaMemcpyDeviceToHost);

    int sum = 0;

    /*for (int i = 0; i < n; ++i) {
        printf("%f ", h_idata[i]);
    }*/
    for (int i = 0; i < n; ++i) {
        printf("%f \n", h_dataY[i]);
        printf("%f \n", h_dataZ[i]);
        printf("%i \n", h_permutations[i]);
        //printf("%f ", h_idata[i]);
        /*if (h_odata[i] > cut) {
            throw std::runtime_error("Partition failed");
        }*/
    }

    printf("\n");

    cudaFree(d_dataX);
    cudaFree(d_dataY);
    cudaFree(d_dataZ);
    cudaFree(d_dataT);
    cudaFree(d_permutations);
    cudaFree(d_offsetGreater);
    cudaFree(d_offsetLessEquals);

    free(h_dataX);
    free(h_dataY);
    free(h_dataZ);

    cudaDeviceReset();
}