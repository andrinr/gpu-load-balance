//
// Created by andrin on 15/05/22.
//
#include <blitz/array.h>

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
__global__ void reduce(float *g_idata, int *g_odata, float cut) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    // todo: Why have such long stride here, no bank conflicts?
    sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
    __syncthreads();

    for (unsigned int s=blockDim.x/2; s>32; s>>=1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

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
    if (tid < 32) {
        warpReduce<blockSize>(sdata, tid);
    }
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

int main(int argc, char** argv) {

    int n = 1 << 20;
    int nd = 3;

    std::cout << n << "\n";

    blitz::Array<float, 2> p = blitz::Array<float, 2>(n, nd);
    srand(0);
    for (int i = 0; i < p.rows(); i++) {
        for (int d = 0; d < nd; d++) {
            p(i,d) = (float)(rand())/(float)(RAND_MAX);
        }
    }

    const int nThreads = 256;
    int nBlocks = n / nThreads / 2;

    float * h_particles = p.data();
    float * d_particles;
    int * d_sums;
    int * h_sums = (int*)malloc(n);
    cudaMalloc(&d_particles, sizeof (float) * n * nd);
    cudaMemcpy(d_particles, h_particles, sizeof (float ) * n * nd, cudaMemcpyHostToDevice);

    cudaMalloc(&d_sums, sizeof (int) * n);

    // Number of threads per block is limited

    // Need for cut service becomes clear here!
    float cut = 0.5;
    reduce<nThreads><<<nBlocks, nThreads>>>(d_particles, d_sums, cut);

    cudaMemcpy(h_sums, d_sums, sizeof (int ) * n, cudaMemcpyDeviceToHost);

    cudaFree(d_particles);
    cudaFree(d_sums);

    cudaDeviceReset();

    std::cout << n << " " << h_sums[0] << "\n";
}
