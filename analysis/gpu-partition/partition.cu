#include <blitz/array.h>
#include "../../src/utils/condReduce.cuh"

// https://www.cse.chalmers.se/~tsigas/papers/GPU-Quicksort-jea.pdf
// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
// https://github.com/NVIDIA/cuda-samples/tree/master/Samples/2_Concepts_and_Techniques/scan
// https://onlinelibrary.wiley.com/doi/epdf/10.1002/cpe.3611

#define SHARED_MEMORY_BANKS 32
#define LOG_MEM_BANKS 4
#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))

// 2 data elements per thread
template <unsigned int blockSize>
__device__ void d_scan(volatile uint * s_idata, uint thid, int n) {

    for (int d = n>>1; d > 0; d >>= 1) { // build sum in place up the tree
        __syncthreads();
        if (thid < d) {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            s_idata[bi] += s_idata[ai];
        }
        offset *= 2;
    }
    if (thid==0) {
        s_idata[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
    }

    for (int d = 1; d < n; d *= 2) {// traverse down tree & build scan
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
__global__ void partition(int totalLeft, int totalRight, uint *  offsetLeq, uint *  offsetG, float * g_idata, float * g_odata, float pivot) {
    extern __shared__ uint s_lqPivot[];
    extern __shared__ uint s_gPivot[];
    extern __shared__ float s_res[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize * 2)+threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;

    uint l_offsetLeq = offsetLeq[blockIdx.x];
    uint l_offsetG = offsetG[blockIdx.x];

    uint leftOffset = atomicAdd(&totalLeft, l_offsetLeq);
    uint rightOffset = atomicAdd(&totalRight, l_offsetG);

    uint f1 = g_idata[2*i] < pivot
    s_lqPivot[2 * tid] = f;
    s_gPivot[2 * tid] = 1-f;

    uint f2 = g_idata[2 * i + 1] < pivot
    s_lqPivot[2 * tid + 1] = f;
    s_gPivot[2 * tid + 1] = 1-f;

    __syncthreads();

    uint offLq = scan(s_lqPivot, tid);
    uint offG = scan(s_gPivot, tid);

    __syncthreads();

    // avoiding branch divergence
    g_odata[
            (s_lqPivot[2*tid] + leftOffset) * f1 +
            (s_gPivot[2*tid] + righOffset) * (1-f1)] = g_idata[2*i];

    g_odata[
            (s_lqPivot[2*tid+1] + leftOffset) * f2 +
            (s_gPivot[2*tid+1] + righOffset) * (1-f2)] = g_idata[2*i+1];

}


int main(int argc, char** argv) {

    const int N_THREADS = 512;
    const int ELEMENTS_PER_THREAD = 32;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int n = 1 << 13;
    float cut = 0.5;

    float * pos = (float*)malloc(n);
    for (int i = 0; i < n; i++) {
        pos[i] = i;
    }

    // Can increase speed by another factor of around two
    int elementsPerThread = 32;
    int nBlocks = (int) ceil((float) n / (N_THREADS * 2.0));
    printf("nThreads %i, nBlocks %i, n %i \n", nThreads, nBlocks, n);

    float * d_idata;
    float * d_odata;
    cudaMalloc(&d_idata, sizeof (float) * n);
    cudaMalloc(&d_odata, sizeof (float) * n);
    cudaMemcpy(d_idata, pos, sizeof (float ) * n, cudaMemcpyHostToDevice);

    uint * countA = (uint*)malloc(nBlocks * sizeof(uint));
    uint * countB = (uint*)malloc(nBlocks * sizeof(uint));

    cudaEventRecord(start);

    conditionalReduce<N_THREADS, true>(
            g_idata,
            countA,
            cut,
            n,
            nBlocks,
            N_THREADS,
            N_THREADS * sizeof (uint) * 2
    );

    conditionalReduce<N_THREADS, false>(
            g_idata,
            countB,
            cut,
            n,
            nBlocks,
            N_THREADS,
            N_THREADS * sizeof (uint) * 2
    );



    partition<nThreads><<<
            nBlocks,
            nThreads,
            nThreads * sizeof (uint) * 2 >>>(
            countA,
            countB,
            g_idata,
            g_odata,
            cut;

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

int ServiceReshuffle::Service(PST pst,void *vin,int nIn,void *vout, int nOut) {

    auto lcl = pst->lcl;
    auto in  = static_cast<input *>(vin);
    auto out = static_cast<output *>(vout);
    const int nCells = nIn / sizeof(input);
    assert(nOut / sizeof(output) >= nCells);

    //int bytes = nCounts * sizeof (uint);
    // https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/
    int blockOffset = 0;
    std::array<int, MAX_CELLS> offsets;
    offsets[0] = 0;

    int countLeq[nCells];
    int countG[nCells];

    for (int cellPtrOffset = 0; cellPtrOffset < nCells; ++cellPtrOffset) {
        out[cellPtrOffset] = 0;
    }

    for (int cellPtrOffset = 0; cellPtrOffset < nCells; ++cellPtrOffset) {
        auto cell = static_cast<Cell>(*(in + cellPtrOffset));

        int beginInd = pst->lcl->cellToRangeMap(cell.id, 0);
        int endInd =  pst->lcl->cellToRangeMap(cell.id, 1);
        int n = endInd - beginInd;
        float cut = cell.getCut();

        const int nBlocks = (int) ceil((float) n / (N_THREADS *  ELEMENTS_PER_THREAD));

        orbitUtils::conditionalReduce<N_THREADS, true>(
                lcl->d_particles + beginInd,
                lcl->d_resultsA + blockOffset,
                cut,
                n,
                nBlocks,
                N_THREADS,
                N_THREADS * sizeof (uint),
                lcl->stream
        );

        orbitUtils::conditionalReduce<N_THREADS, false>(
                lcl->d_particles + beginInd,
                lcl->d_resultsB + blockOffset,
                cut,
                n,
                nBlocks,
                N_THREADS,
                N_THREADS * sizeof (uint),
                lcl->stream
        );

        offsets[cellPtrOffset+1] = blockOffset;
    }

    CUDA_CHECK(cudaMemcpyAsync,(
            lcl->h_resultsA,
                    lcl->d_resultsA,
                    sizeof (uint) * blockOffset,
                    cudaMemcpyDeviceToHost,
                    lcl->stream));

    CUDA_CHECK(cudaMemcpyAsync,(
            lcl->h_resultsB,
                    lcl->d_resultsB,
                    sizeof (uint) * blockOffset,
                    cudaMemcpyDeviceToHost,
                    lcl->stream));

    CUDA_CHECK(cudaStreamSynchronize,(lcl->stream));

    int totalLeqOffset = 0;
    int totalGOffset = 0;
    for (int cellPtrOffset = 0; cellPtrOffset < nCells; ++cellPtrOffset) {
        int begin = offsets[cellPtrOffset];
        int end = offsets[cellPtrOffset + 1];

        for (int i = begin; i < end; ++i) {
            totalLeqOffset += lcl->h_resultsA[i];
            totalGOffset += lcl->h_resultsB[i];
        }

        const int nBlocks = (int) ceil((float) n / (N_THREADS *  ELEMENTS_PER_THREAD));

        auto cell = static_cast<Cell>(*(in + cellPtrOffset));

        int beginInd = pst->lcl->cellToRangeMap(cell.id, 0);
        int endInd =  pst->lcl->cellToRangeMap(cell.id, 1);

        // todo: pay attention to axis
        pivotPrefixSum<N_THREADS>
        <<<nBlocks,
                N_THREADS,
                N_THREADS * sizeof (uint) * 2 + N_THREADS * sizeof (float),
                lcl->stream>>>(
                totalLeqOffset,
                        totalGOffset,
                        lcl->d_particles + beginInd,
                        cell.getCut()
        );
    }

    CUDA_CHECK(cudaStreamSynchronize,(lcl->stream));
    return 0;
}




