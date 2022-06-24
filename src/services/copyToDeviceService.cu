#include "copyToDeviceService.h"
#include <blitz/array.h>
#include <vector>

// Make sure that the communication structure is "trivial" so that it
// can be moved around with "memcpy" which is required for MDL.
static_assert(std::is_void<ServiceCopyToDevice::input>()  || std::is_trivial<ServiceCopyToDevice::input>());
static_assert(std::is_void<ServiceCopyToDevice::output>() || std::is_trivial<ServiceCopyToDevice::output>());

int ServiceCopyToDevice::Service(PST pst,void *vin,int nIn,void *vout, int nOut) {
    // store streams / initialize in local data
    auto lcl = pst->lcl;

    int nParticles = lcl->particles.rows();
    // We only need the first nParticles, since axis 0 is axis where cuts need to be found
    cudaMemcpyAsync(
            lcl->d_particles,
            lcl->particles.data(),
            sizeof (float) * nParticles,
            cudaMemcpyHostToDevice,
            pst->lcl->stream
    );

    return sizeof(output);
}

int ServiceCopyToDevice::Combine(void *vout,void *vout2,int nIn,int nOut1,int nOut2) {

    return 0;
}
