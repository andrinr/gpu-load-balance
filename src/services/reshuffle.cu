#include "reshuffle.cuh"
#include "../cell.h"
#include <blitz/array.h>

#include <cooperative_groups.h>

namespace cg = cooperative_groups;
// Make sure that the communication structure is "trivial" so that it
// can be moved around with "memcpy" which is required for MDL.
static_assert(std::is_void<ServiceReshuffle::input>()  || std::is_trivial<ServiceReshuffle::input>());
static_assert(std::is_void<ServiceReshuffle::output>() || std::is_trivial<ServiceReshuffle::output>());

// Exclusive vector scan: the array to be scanned is stored
// in local thread memory scope as uint4
inline __device__ uint scan1Inclusive(uint idata, volatile uint *s_Data,
                                      uint size, cg::thread_block cta) {
    uint pos = 2 * threadIdx.x - (threadIdx.x & (size - 1));
    s_Data[pos] = 0;
    pos += size;
    s_Data[pos] = idata;

    for (uint offset = 1; offset < size; offset <<= 1) {
        cg::sync(cta);
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

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize)+threadIdx.x*4;
    unsigned int gridSize = blockSize*gridDim.x;

    uint f = g_idata[i] < pivot
    s_data[tid].x = f;
    s_data[tid].y = 1-f;

    __syncthreads();


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
