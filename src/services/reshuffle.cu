#include "reshuffle.cuh"
#include "../cell.h"
#include <blitz/array.h>
#include "../utils/condReduce.cuh"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
// Make sure that the communication structure is "trivial" so that it
// can be moved around with "memcpy" which is required for MDL.
static_assert(std::is_void<ServiceReshuffle::input>()  || std::is_trivial<ServiceReshuffle::input>());
static_assert(std::is_void<ServiceReshuffle::output>() || std::is_trivial<ServiceReshuffle::output>());

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
    }

    if (j < n) {
        g_odata[indexB] = g_idata[j];
    }
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

int ServiceReshuffle::Combine(void *vout,void *vout2,int nIn,int nOut1,int nOut2) {
    return 0;
}
