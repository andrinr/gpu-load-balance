#include "initGPUService.h"
#include <blitz/array.h>

// Make sure that the communication structure is "trivial" so that it
// can be moved around with "memcpy" which is required for MDL.
static_assert(std::is_void<ServiceInitGPU::input>()  || std::is_trivial<ServiceInitGPU::input>());
static_assert(std::is_void<ServiceInitGPU::output>() || std::is_trivial<ServiceInitGPU::output>());

int ServiceInitGPU::Service(PST pst,void *vin,int nIn,void *vout, int nOut) {

    auto lcl = pst->lcl;
    ServiceInitGPU::input in = *static_cast<input *>(vin);

    cudaStreamCreate(&lcl->stream);

    return 0;
}

int ServiceInitGPU::Combine(void *vout,void *vout2,int nIn,int nOut1,int nOut2) {
    return 0;
}
