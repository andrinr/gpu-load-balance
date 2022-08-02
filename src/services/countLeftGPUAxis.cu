#include "countLeftGPUAxis.h"
#include <blitz/array.h>
#include <array>

// Make sure that the communication structure is "trivial" so that it
// can be moved around with "memcpy" which is required for MDL.
static_assert(std::is_void<ServiceCountLeftGPUAxis::input>()  || std::is_trivial<ServiceCountLeftGPUAxis::input>());
static_assert(std::is_void<ServiceCountLeftGPUAxis::output>() || std::is_trivial<ServiceCountLeftGPUAxis::output>());

template <unsigned int blockSize>
extern __device__ void warpReduce(volatile unsigned int *s_data, unsigned int tid) {
    if (blockSize >= 64) s_data[tid] += s_data[tid + 32];
    if (blockSize >= 32) s_data[tid] += s_data[tid + 16];
    if (blockSize >= 16) s_data[tid] += s_data[tid + 8];
    if (blockSize >= 8) s_data[tid] += s_data[tid + 4];
    if (blockSize >= 4) s_data[tid] += s_data[tid + 2];
    if (blockSize >= 2) s_data[tid] += s_data[tid + 1];
}

template <unsigned int blockSize>
extern __global__ void reduce(
        float * g_idata,
        unsigned int * g_begins,
        unsigned int * g_ends,
        float * g_cuts,
        unsigned int * g_odata) {

    __shared__ unsigned int s_data[blockSize];

    unsigned int tid = threadIdx.x;
    const unsigned int begin = g_begins[blockIdx.x];
    const unsigned int end = g_ends[blockIdx.x];
    const float cut = g_cuts[blockIdx.x];

    unsigned int i = begin + tid;
    s_data[tid] = 0;

    // unaligned coalesced g memory access
    while (i < end) {
        s_data[tid] += (g_idata[i] <= cut);
        i += blockSize;
    }
    __syncthreads();

    if (blockSize >= 512) {
        if (tid < 256) {
            s_data[tid] += s_data[tid + 256];
        }
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) {
            s_data[tid] += s_data[tid + 128];
        } __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) {
            s_data[tid] += s_data[tid + 64];
        } __syncthreads();
    }
    if (tid < 32) {
        warpReduce<blockSize>(s_data, tid);
    }
    if (tid == 0) {
        g_odata[blockIdx.x] = s_data[0];
    }
}

int ServiceCountLeftGPUAxis::Service(PST pst,void *vin,int nIn,void *vout, int nOut) {
    // store streams / initialize in local d
    // ata
    auto lcl = pst->lcl;
    auto in  = static_cast<input *>(vin);
    auto out = static_cast<output *>(vout);
    unsigned int nCells = nIn / sizeof(input);
    assert(nOut / sizeof(output) >= nCells);

    std::vector<unsigned  int> cellIndices ;

    int nBlocks = 0;
    int blockPtr = 0;
    for (int cellPtrOffset = 0; cellPtrOffset < nCells; ++cellPtrOffset) {
        auto cell = static_cast<Cell>(*(in + cellPtrOffset));
        unsigned int beginInd = pst->lcl->cellToRangeMap(cell.id, 0);
        unsigned int endInd =  pst->lcl->cellToRangeMap(cell.id, 1);
        unsigned int n = endInd - beginInd;

        unsigned int nBlocksPerCell = (int) floor((float) n / (N_THREADS * ELEMENTS_PER_THREAD));

        int begin = beginInd;
        for (int i = 0; i < nBlocksPerCell; ++i) {
            lcl->h_cuts[blockPtr] = cell.getCut();
            cellIndices.push_back(cellPtrOffset);
            blockPtr++;
        }
        nBlocks += nBlocksPerCell;

        out[cellPtrOffset] = 0;
    }

    CUDA_CHECK(cudaMemcpyAsync,(
            lcl->d_cuts,
            lcl->h_cuts,
            sizeof (float) * nBlocks,
            cudaMemcpyHostToDevice,
            lcl->streams(0)));

    //CUDA_CHECK(cudaMemset, (lcl->d_index, 0, sizeof(unsigned int)));

    // Execute the kernel
    reduce<N_THREADS><<<
            nBlocks,
            N_THREADS,
            N_THREADS * sizeof (unsigned int),
            lcl->streams(0)
            >>>(
                lcl->d_particlesT,
                lcl->d_begins,
                lcl->d_ends,
                lcl->d_cuts,
                lcl->d_results
            );

    //
    CUDA_CHECK(cudaMemcpyAsync,(
            lcl->h_results,
            lcl->d_results,
            sizeof (unsigned int) * nBlocks,
            cudaMemcpyDeviceToHost,
            lcl->streams(0)
    ));

    CUDA_CHECK(cudaStreamSynchronize,(lcl->streams(0)));

    for (int i = 0; i < nBlocks; ++i) {
        out[cellIndices[i]] += lcl->h_results[i];
    }

    return nCells * sizeof(output);
}

int ServiceCountLeftGPUAxis::Combine(void *vout,void *vout2,int nIn,int nOut1,int nOut2) {
    auto out  = static_cast<output *>(vout);
    auto out2 = static_cast<output *>(vout2);
    int nCounts = nIn / sizeof(input);
    assert(nOut1 >= nCounts*sizeof(output));
    assert(nOut2 >= nCounts*sizeof(output));
    for(auto i=0; i<nCounts; ++i)
	    out[i] += out2[i];
    return nCounts * sizeof(output);
}
