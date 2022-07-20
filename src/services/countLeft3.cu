#include "countLefGPU.h"
#include <blitz/array.h>
#include <array>
#include "../utils/condReduce.cuh"

// Make sure that the communication structure is "trivial" so that it
// can be moved around with "memcpy" which is required for MDL.
static_assert(std::is_void<ServiceCountLeftGPU::input>()  || std::is_trivial<ServiceCountLeftGPU::input>());
static_assert(std::is_void<ServiceCountLeftGPU::output>() || std::is_trivial<ServiceCountLeftGPU::output>());

template <unsigned int blockSize>
extern __device__ void warpReduce(volatile int *sdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
extern __global__ void reduce(
        float * g_idata,
        unsigned int * g_begins,
        unsigned int * g_ends,
        float * cuts,
        unsigned int * a_index,
        unsigned int * g_odata) {

    extern __shared__ int sdata[];
    __shared__ unsigned int index;

    unsigned int tid = threadIdx.x;

    if (tid == 0) {
        index = atomicAdd(a_index, 1);
    }

    int start = g_begins[index];
    int end = g_ends[index];
    int n = end - start;

    unsigned int i = start + threadIdx.x;
    unsigned int gridSize = blockSize*gridDim.x;
    sdata[tid] = 0;

    while (i < n) {
        sdata[tid] += (g_idata[i] <= cut);
        i += gridSize;
    }
    __syncthreads();

    if (blockSize >= 512) {
        if (tid < 256) {
            sdata[tid] += sdata[tid + 256];
        }
        __syncthreads();
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
        g_odata[index] = sdata[0];
    }
}

int ServiceCountLeftGPU::Service(PST pst,void *vin,int nIn,void *vout, int nOut) {
    // store streams / initialize in local d
    // ata
    auto lcl = pst->lcl;
    auto in  = static_cast<input *>(vin);
    auto out = static_cast<output *>(vout);
    const int nCells = nIn / sizeof(input);
    assert(nOut / sizeof(output) >= nCells);

    std::vector<float> cellIDs;

    int nBlocks = 0;
    for (int cellPtrOffset = 0; cellPtrOffset < nCells; ++cellPtrOffset) {
        auto cell = static_cast<Cell>(*(in + cellPtrOffset));
        int beginInd = pst->lcl->cellToRangeMap(cell.id, 0);
        int endInd =  pst->lcl->cellToRangeMap(cell.id, 1);
        int n = endInd - beginInd;

        const int nBlocksPerCell = (int) ceil((float) n / (N_THREADS * ELEMENTS_PER_THREAD));
        nBlocks += nBlocksPerCell;

        int begin = beginInd;
        for (int i = 0; i < nBlocksPerCell; ++i) {
            lcl->h_cuts[i] = cell.getCut();
            lcl->h_begins[i] = begin;
            begin += N_THREADS * ELEMENTS_PER_THREAD;
            lcl->h_ends = min(begin, endInd);
            cellIDs.push_back(cell.id);
        }
    }

    CUDA_CHECK(cudaMemcpyAsync,(
            lcl->h_begins,
            lcl->d_begins,
            sizeof (unsigned int) * nBlocks,
            cudaMemcpyDeviceToHost,
            lcl->streams(0)));

    CUDA_CHECK(cudaMemcpyAsync,(
            lcl->h_ends,
            lcl->d_ends,
            sizeof (unsigned int) * nBlocks,
            cudaMemcpyDeviceToHost,
            lcl->streams(0)));

    CUDA_CHECK(cudaMemcpyAsync,(
            lcl->h_cuts,
            lcl->d_cuts,
            sizeof (float) * nBlocks,
            cudaMemcpyDeviceToHost,
            lcl->streams(0)));

    CUDA_CHECK(cudaMemset, (lcl->d_index, 0, sizeof(unsigned int)));

    // Execute the kernel
    reduce<N_THREADS><<<
            N_THREADS,
            nBlocks,
            N_THREADS * 2 * sizeof (unsigned int)
            >>>(
                lcl->d_particlesT,
                lcl->d_begins,
                lcl->d_ends,
                lcl->d_cuts,
                lcl->d_index,
                lcl->d_results
                    );

    CUDA_CHECK(cudaMemcpyAsync,(
            lcl->h_results,
            lcl->d_results,
            sizeof (unsigned int) * blockOffset,
            cudaMemcpyDeviceToHost,
            lcl->streams(0)));


    for (int i = 0; i < nBlocks; ++i) {
        out[cellIDs[i]] += lcl->h_results[i];
    }

    return nCells * sizeof(output);
}

int ServiceCountLeftGPU::Combine(void *vout,void *vout2,int nIn,int nOut1,int nOut2) {
    auto out  = static_cast<output *>(vout);
    auto out2 = static_cast<output *>(vout2);
    int nCounts = nIn / sizeof(input);
    assert(nOut1 >= nCounts*sizeof(output));
    assert(nOut2 >= nCounts*sizeof(output));
    for(auto i=0; i<nCounts; ++i)
	    out[i] += out2[i];
    return nCounts * sizeof(output);
}
