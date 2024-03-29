#include "countLeftGPUAxis.h"
#include <blitz/array.h>
#include <array>

// Make sure that the communication structure is "trivial" so that it
// can be moved around with "memcpy" which is required for MDL.
static_assert(std::is_void<ServiceCountLeftGPUAxis::input>()  || std::is_trivial<ServiceCountLeftGPUAxis::input>());
static_assert(std::is_void<ServiceCountLeftGPUAxis::output>() || std::is_trivial<ServiceCountLeftGPUAxis::output>());

#define FULL_MASK 0xffffffff
template <unsigned int blockSize>
extern __device__ void warpReduce2(volatile unsigned int *s_data, unsigned int tid) {
    if (blockSize >= 64) s_data[tid] += s_data[tid + 32];
    __syncwarp();
    if (tid < 32) {
        unsigned int val = s_data[tid];
        for (int offset = 16; offset > 0; offset /= 2)
            val += __shfl_down_sync(FULL_MASK, val, offset);
        s_data[tid] = val;
    }
}

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

template <unsigned int blockSize>
extern __global__ void reduce2(
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

    unsigned int val = 0;
    // unaligned coalesced g memory access
    while (i < end) {
        val += (g_idata[i] <= cut);
        i += blockSize;
    }
    __syncwarp();

    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(FULL_MASK, val, offset);

    s_data[tid] = val;
    __syncthreads();

    if (tid > 32 ) {
        return;
    }

    val = 0;
    if (tid < 32 && tid * 32 < blockSize) {
        val += s_data[tid * 32];
    }
    __syncwarp();

    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(FULL_MASK, val, offset);

    if (tid == 0) {
        g_odata[blockIdx.x] = val;
    }
}

template <unsigned int blockSize>
extern __global__ void reduce3(
        float * g_idata,
        unsigned int * g_begins,
        unsigned int * g_ends,
        float * g_cuts,
        unsigned int * g_odata) {

    static __shared__ unsigned int s_data[blockSize];
    static __shared__ unsigned int begin;
    static __shared__ unsigned int end;
    static __shared__ float cut;

    unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & (32 - 1);
    const unsigned int warpId = tid >> 5;

    if (tid == 0) {
        begin = g_begins[blockIdx.x];
        end = g_ends[blockIdx.x];
        cut = g_cuts[blockIdx.x];
    }
    __syncthreads();
    unsigned int i = begin + tid;

    unsigned int val = 0;
    // unaligned coalesced g memory access
    while (i < end) {
        val += (g_idata[i] <= cut);
        i += blockSize;
    }
    __syncwarp();

    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }

    if (lane == 0) {
        s_data[warpId] = val;
    }
    __syncthreads();

    // All warps but first one are not needed anymore
    if (warpId == 0) {
        val = (tid < blockDim.x / warpSize) ? s_data[lane] : 0;

        for (int offset = blockSize / 32 ; offset > 0; offset /= 2) {
            val += __shfl_down_sync(FULL_MASK, val, offset);
        }

        if (tid == 0) {
            g_odata[blockIdx.x] = val;
        }
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

        unsigned int nBlocksPerCell = max((int) floor((float) n / (N_THREADS * ELEMENTS_PER_THREAD)),1);

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
    reduce3<N_THREADS><<<
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
