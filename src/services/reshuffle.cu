#include "reshuffle.cuh"
#include "../cell.h"
#include <blitz/array.h>
#include "../utils/reduce.cuh"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
// Make sure that the communication structure is "trivial" so that it
// can be moved around with "memcpy" which is required for MDL.
static_assert(std::is_void<ServiceReshuffle::input>()  || std::is_trivial<ServiceReshuffle::input>());
static_assert(std::is_void<ServiceReshuffle::output>() || std::is_trivial<ServiceReshuffle::output>());



template <unsigned int blockSize>
inline __device__ uint scan(volatile uint s_idata, uint offset) {
    uint pos = 2 * threadIdx.x - (threadIdx.x & (size - 1));
    s_idata[pos] = 0;
    pos += size;
    s_idata[pos] = idata;

    for (uint offset = 1; offset < size; offset <<= 1) {
        __syncthreads();
        uint t = s_Data[pos] + s_Data[pos - offset];
        cg::sync(cta);
        s_Data[pos] = t;
    }

    return s_Data[pos];
}

template <unsigned int blockSize>
__global__ void pivotPrefixSum( float * g_idata, float * g_odata, float pivot) {
    extern __shared__ uint s_lqPivot[];
    extern __shared__ uint s_gPivot[];
    extern __shared__ float s_res[];

    unsigned int tid = threadIdx.x;
    unsigned int iStart = blockIdx.x * blockSize;
    unsigned int i = blockIdx.x*(blockSize)+threadIdx.x;
    unsigned int gridSize = blockSize*gridDim.x;

    uint f = g_idata[i] < pivot
    s_lqPivot[tid].x = f;
    s_gPivot[tid].y = 1-f;

    __syncthreads();

    uint offLq = scan(s_lqPivot[tid], 0);
    uint offG = scan(s_gPivot[tid], 0);

    __syncthreads();

    s_res[is_lqPivot[tid] + iStart] = g_idata[i];
    s_res[is_lqPivot[tid] + offLq +  iStart] = g_idata[i];

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

    for (int cellPtrOffset = 0; cellPtrOffset < nCells; ++cellPtrOffset) {
        int begin = offsets[cellPtrOffset];
        int end = offsets[cellPtrOffset + 1];

        for (int i = begin; i < end; ++i) {
            countLeq[cellPtrOffset] += lcl->h_resultsA[i];
            countG[cellPtrOffset] += lcl->h_resultsB[i];
        }
    }



    return 0;
}

int ServiceReshuffle::Combine(void *vout,void *vout2,int nIn,int nOut1,int nOut2) {
    return 0;
}
