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
__global__ void reduce(float *g_idata, int *g_odata, float cut, int n) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = 0;
    while (i < n) {
        sdata[tid] += (g_idata[i] < cut) + (g_idata[i+blockSize] < cut);
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
    if (tid < 32) {
        warpReduce<blockSize>(sdata, tid);
    }
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

int main(int argc, char** argv) {

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int n = 1 << 13;
    //int nd = 3;

    float * pos = (float*)calloc(n, sizeof(float));
    for (int i = 0; i < n; i++) {
        pos[i] = (float)(rand())/(float)(RAND_MAX);
    }

    int testSum = 0;
    for (int i = 0; i < n; i++) {
        testSum += pos[i] < 0.5;
    }
    std::cout << testSum << "\n";

    const int nThreads = 512;
    // Can increase speed by another factor of around two
    int elementsPerThread = 2;
    int nBlocks = (int) ceil((float) n / (nThreads * 2.0 * elementsPerThread));
    printf("nThreads %i, nBlocks %i, n %i \n", nThreads, nBlocks, n);


    float * d_particles;
    int * d_sums;
    int * h_sums = (int*)calloc(n, sizeof(int));
    cudaMalloc(&d_particles, sizeof (float) * n);
    cudaMemcpy(d_particles, pos, sizeof (float ) * n, cudaMemcpyHostToDevice);

    cudaMalloc(&d_sums, sizeof (int) * n);

    float cut = 0.5;

    cudaEventRecord(start);
    reduce<nThreads><<<nBlocks, nThreads, nThreads * sizeof (int) >>>(d_particles, d_sums, cut, n);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << milliseconds << "\n";

    cudaMemcpy(h_sums, d_sums, sizeof (int ) * nBlocks, cudaMemcpyDeviceToHost);

    int sum = 0;

    for (int i = 0; i < nBlocks; ++i) {
        printf("sum %i \n", sum);
        sum += h_sums[i];
    }

    std::cout << "is " << sum << " should be " << n / 2 << " \n";

    cudaFree(d_particles);
    cudaFree(d_sums);

    free(h_sums);
    free(pos);

    cudaDeviceReset();
}
