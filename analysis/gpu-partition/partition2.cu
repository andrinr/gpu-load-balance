#include <blitz/array.h>
#include "../../src/utils/condReduce.cuh"
#include "../../src/constants.h"
#include <thrust/partition.h>
// https://www.cse.chalmers.se/~tsigas/papers/GPU-Quicksort-jea.pdf
// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
// https://github.com/NVIDIA/cuda-samples/tree/master/Samples/2_Concepts_and_Techniques/scan
// https://onlinelibrary.wiley.com/doi/epdf/10.1002/cpe.3611

struct is_smaller_cut
{
    float pivot;
    is_smaller_cut(float pivot) : pivot(pivot) {};

    __host__ __device__
    bool operator()(const float &x)
    {
        return (x < pivot) == 0;
    }
};

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
    CUDA_CHECK(cudaMemcpy,(d_idata, pos, sizeof (float ) * n, cudaMemcpyHostToDevice));

    unsigned int * d_totalLeq;
    unsigned int * d_totalG;

    CUDA_CHECK(cudaMalloc,(&d_totalLeq, sizeof(uint)));
    CUDA_CHECK(cudaMalloc,(&d_totalG, sizeof(uint)));

    CUDA_CHECK(cudaMemset,(&d_totalLeq, 0, sizeof(uint)));
    CUDA_CHECK(cudaMemset,(&d_totalG, 0, sizeof(uint)));

    cudaEventRecord(start);

    const int N = sizeof(pos)/sizeof(int);
    thrust::partition(thrust::host,
                      pos, pos + N,
                      is_smaller_cut(cut));

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