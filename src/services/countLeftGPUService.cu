#include "countLeftService.h"
#include <blitz/array.h>

// Make sure that the communication structure is "trivial" so that it
// can be moved around with "memcpy" which is required for MDL.
static_assert(std::is_void<ServiceCountLeft::input>()  || std::is_trivial<ServiceCountLeft::input>());
static_assert(std::is_void<ServiceCountLeft::output>() || std::is_trivial<ServiceCountLeft::output>());

template <unsigned int blockSize>
__device__ void warpReduce(volatile int *sdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void reduce(float *g_idata, int *g_odata, float cut, int n) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;
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

    int n = 1 << 27;
    int nd = 3;

    float * pos = (float*)calloc(n, sizeof(float));
    for (int i = 0; i < n; i++) {
        pos[i] = (float)(rand())/(float)(RAND_MAX);
    }

    int testSum = 0;
    for (int i = 0; i < 10000; i++) {
        testSum += pos[i] < 0.5;
    }
    std::cout << testSum << "\n";

    const int nThreads = 256;
    // Can increase speed by another factor of around two
    int elementsPerThread = 16;
    int nBlocks = ceil(n / nThreads / 2 / elementsPerThread);

    float * d_particles;
    int * d_sums;
    int * h_sums = (int*)calloc(n, sizeof(int));
    cudaMalloc(&d_particles, sizeof (float) * n);
    cudaMemcpy(d_particles, pos, sizeof (float ) * n, cudaMemcpyHostToDevice);

    cudaMalloc(&d_sums, sizeof (int) * n);

    // Number of threads per block is limited

    // Need for cut service becomes clear here!
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
        sum += h_sums[i];
    }

    std::cout << sum << " " << n << "\n";

    cudaFree(d_particles);
    cudaFree(d_sums);

    free(h_sums);
    free(pos);

    cudaDeviceReset();
}


int ServiceCountLeft::Service(PST pst,void *vin,int nIn,void *vout, int nOut) {
    auto lcl = pst->lcl;
    auto in  = static_cast<input *>(vin);
    auto out = static_cast<output *>(vout);
    auto nCells = nIn / sizeof(input);
    assert(nOut / sizeof(output) >= nCells);
    printf("ServiceCountLeft invoked on thread %d\n",pst->idSelf);

    cudaMalloc(&d_particles, sizeof (float) * n);
    cudaMemcpy(d_particles, pos, sizeof (float ) * n, cudaMemcpyHostToDevice);

    cudaMalloc(&d_sums, sizeof (int) * n);

    // https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/
    for (int cellPtrOffset = 0; cellPtrOffset < nCells; ++cellPtrOffset){

        auto cell = static_cast<Cell>*(in + cellPtrOffset);
        // -1 axis signals no need to count
        if (cell.cutAxis == -1) continue;
        int beginInd = pst->lcl->cellToRangeMap(cell.id, 0);
        int endInd =  pst->lcl->cellToRangeMap(cell.id, 1);

        const int nThreads = 256;
        // Can increase speed by another factor of around two
        const int elementsPerThread = 16;
        const int nBlocks = ceil(endInd - beginInd / (nThreads * elementsPerThread) / 2 );

        cudaStream_t stream;
        cudaError_t result;
        result = cudaStreamCreate(&stream);

        float * d_particles;
        int * d_counts;
        int * h_counts = (int*)calloc(nBlocks, sizeof(int));

        blitz::Array<float,1> particles = pst->lcl->particles(blitz::Range(beginInd, endInd), cell.cutAxis);

        cudaMalloc(&d_particles, sizeof (float) * n);
        cudaMalloc(&d_counts, sizeof (int) * nBlocks);
        result = cudaMemcpyAsync(d_particles, particles.data(), endInd - beginInd, cudaMemcpyHostToDevice, stream);

        float cut = (cell.cutMarginRight + cell.cutMarginLeft) / 2.0;
        reduce<nThreads><<<nBlocks, nThreads, nThreads * sizeof (int), stream>>>(
                d_particles, d_counts, cut, endInd - beginInd);

        cudaMemcpy(h_sums, d_sums, sizeof (int ) * nBlocks, cudaMemcpyDeviceToHost);

        for (int i = 0; i < nBlocks; ++i) {
            out[cellPtrOffset] += h_sums[i];
        }
    }

    // Wait till all streams have finished
    cudaDeviceSynchronize();

    return nCells * sizeof(output);
}

int ServiceCountLeft::Combine(void *vout,void *vout2,int nIn,int nOut1,int nOut2) {
    auto out  = static_cast<output *>(vout);
    auto out2 = static_cast<output *>(vout2);
    int nCounts = nIn / sizeof(input);
    assert(nOut1 >= nCounts*sizeof(output));
    assert(nOut2 >= nCounts*sizeof(output));
    for(auto i=0; i<nCounts; ++i)
	    out[i] += out2[i];
    return nCounts * sizeof(output);
}
