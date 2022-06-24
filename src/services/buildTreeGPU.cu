#include "buildTreeGPU.h"
#include <blitz/array.h>
#include <vector>

// Make sure that the communication structure is "trivial" so that it
// can be moved around with "memcpy" which is required for MDL.
static_assert(std::is_void<ServiceCopyToDevice::input>()  || std::is_trivial<ServiceCopyToDevice::input>());
static_assert(std::is_void<ServiceCopyToDevice::output>() || std::is_trivial<ServiceCopyToDevice::output>());

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
__global__ void reduce(float *g_idata, uint *g_odata, float cut, int n) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + threadIdx.x;
    unsigned int gridSize = blockSize * 2 * gridDim.x;
    sdata[tid] = 0;
    // todo: ask doug
    while (i < n) {
        sdata[tid] += (g_idata[i] < cut);

        if (i + blockSize < n) {
            sdata[tid] += (g_idata[i + blockSize] < cut);
        }
        i += gridSize;
    }
    __syncthreads();

    if (blockSize >= 512) {
        if (tid < 256) {
            sdata[tid] += sdata[tid + 256];
        }
        if (blockSize >= 256) {
            if (tid < 128) {
                sdata[tid] += sdata[tid + 128];
            }
            __syncthreads();
        }
        if (blockSize >= 128) {
            if (tid < 64) {
                sdata[tid] += sdata[tid + 64];
            }
            __syncthreads();
        }
        if (tid < 32) {
            warpReduce<blockSize>(sdata, tid);
        }
        if (tid == 0) {
            g_odata[blockIdx.x] = sdata[0];
        }
    }
}

__global__ void build(float *g_idata, uint *g_odata) {
    extern __shared__ sTree[];

}

int ServiceBuildTreeGPU::Service(PST pst,void *vin,int nIn,void *vout, int nOut) {
    // store streams / initialize in local data
    auto lcl = pst->lcl;



    return sizeof(output);
}

int ServiceBuildTreeGPU::Combine(void *vout,void *vout2,int nIn,int nOut1,int nOut2) {

    return 0;
}
