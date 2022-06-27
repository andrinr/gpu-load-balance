#include "reshuffle.cuh"
#include "../cell.h"
#include <blitz/array.h>
// Make sure that the communication structure is "trivial" so that it
// can be moved around with "memcpy" which is required for MDL.
static_assert(std::is_void<ServiceReshuffle::input>()  || std::is_trivial<ServiceReshuffle::input>());
static_assert(std::is_void<ServiceReshuffle::output>() || std::is_trivial<ServiceReshuffle::output>());

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
__global__ void pivotPrefixSum( float * g_idata, float * g_data, float pivot) {
    extern __shared__ int s_data[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize+threadIdx.x;
    unsigned int gridSize = blockSize*gridDim.x;

    s_data[tid] = g_idata[i] < pivot;

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
        g_odata[blockIdx.x] = sdata[0];
    }
}

int ServiceReshuffle::Service(PST pst,void *vin,int nIn,void *vout, int nOut) {

    auto lcl = pst->lcl;
    auto in  = static_cast<input *>(vin);
    auto nCells = nIn / sizeof(input);


    return 0;
}

int ServiceReshuffle::Combine(void *vout,void *vout2,int nIn,int nOut1,int nOut2) {
    return 0;
}
