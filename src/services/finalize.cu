#include "finalize.cuh"
#include <blitz/array.h>
#include <vector>

// Make sure that the communication structure is "trivial" so that it
// can be moved around with "memcpy" which is required for MDL.
static_assert(std::is_void<ServiceFreeDevice::input>()  || std::is_trivial<ServiceFreeDevice::input>());
static_assert(std::is_void<ServiceFreeDevice::output>() || std::is_trivial<ServiceFreeDevice::output>());

int ServiceFreeDevice::Service(PST pst,void *vin,int nIn,void *vout, int nOut) {
    // store streams / initialize in local data
    auto lcl = pst->lcl;

    cudaFree(lcl->d_particles);
    cudaFree(lcl->d_counts);
    cudaFreeHost(lcl->particlesAxis.data());

    cudaFreeHost(lcl->h_counts);

    return sizeof(output);
}

int ServiceFreeDevice::Combine(void *vout,void *vout2,int nIn,int nOut1,int nOut2) {

    return 0;
}
