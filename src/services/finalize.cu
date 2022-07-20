#include "finalize.h"
#include <blitz/array.h>
#include <vector>

// Make sure that the communication structure is "trivial" so that it
// can be moved around with "memcpy" which is required for MDL.
static_assert(std::is_void<ServiceFinalize::input>()  || std::is_trivial<ServiceFinalize::input>());
static_assert(std::is_void<ServiceFinalize::output>() || std::is_trivial<ServiceFinalize::output>());

int ServiceFinalize::Service(PST pst,void *vin,int nIn,void *vout, int nOut) {
    // store streams / initialize in local data
    auto lcl = pst->lcl;
    ServiceFinalize::input in = *static_cast<input *>(vin);

    if (in.params.GPU_COUNT) {
        cudaFreeHost(lcl->particlesT.data());

        cudaFree(lcl->d_particlesT);
        cudaFree(lcl->d_results);
        cudaFreeHost(lcl->h_results);
    }

    if (in.params.GPU_COUNT_ATOMIC) {
        cudaFreeHost(lcl->h_cuts);
        cudaFreeHost(lcl->h_begins);
        cudaFreeHost(lcl->h_ends);
        cudaFree(lcl->d_cuts);
        cudaFree(lcl->d_begins);
        cudaFree(lcl->d_ends);
        cudaFree(lcl->d_index);
    }

    if (in.params.GPU_PARTITION) {
        cudaFree(lcl->d_particlesX);
        cudaFree(lcl->d_particlesY);
        cudaFree(lcl->d_particlesZ);
        cudaFree(lcl->d_offsetLeq);
        cudaFree(lcl->d_offsetG);
    }


    return sizeof(output);
}

int ServiceFinalize::Combine(void *vout,void *vout2,int nIn,int nOut1,int nOut2) {

    return 0;
}
