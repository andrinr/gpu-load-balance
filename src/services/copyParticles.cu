#include "copyParticles.h"
#include <blitz/array.h>
#include <vector>
#include "../constants.h"
// Make sure that the communication structure is "trivial" so that it
// can be moved around with "memcpy" which is required for MDL.
static_assert(std::is_void<ServiceCopyParticles::input>()  || std::is_trivial<ServiceCopyParticles::input>());
static_assert(std::is_void<ServiceCopyParticles::output>() || std::is_trivial<ServiceCopyParticles::output>());

int ServiceCopyParticles::Service(PST pst,void *vin,int nIn,void *vout, int nOut) {
    // store streams / initialize in local data
    auto lcl = pst->lcl;
    ServiceCopyParticles::input in = *static_cast<input *>(vin);

    int nParticles = lcl->particles.rows();
    // We only need the first nParticles, since axis 0 is axis where cuts need to be found

    printf("nParticles: %d\n", nParticles);
    if (in.params.GPU_COUNT and not in.params.GPU_PARTITION) {
        cudaMemcpyAsync(
                lcl->d_particlesT,
                lcl->particlesT.data(),
                sizeof (float) * nParticles,
                cudaMemcpyHostToDevice,
                pst->lcl->streams(0)
        );
    }

    if (in.params.GPU_PARTITION) {
        blitz::Array<float, 1> x = lcl->particles(blitz::Range::all(), 0);
        blitz::Array<float, 1> y = lcl->particles(blitz::Range::all(), 1);
        blitz::Array<float, 1> z = lcl->particles(blitz::Range::all(), 2);

        CUDA_CHECK(cudaMemcpyAsync,(
                lcl->d_particlesX,
                x.data(),
                sizeof (float) * nParticles,
                cudaMemcpyHostToDevice,
                pst->lcl->streams(0)
        ));

        CUDA_CHECK(cudaMemcpyAsync,(
                lcl->d_particlesY,
                y.data(),
                sizeof (float) * nParticles,
                cudaMemcpyHostToDevice,
                pst->lcl->streams(0)
        ));

        CUDA_CHECK(cudaMemcpyAsync,(
                lcl->d_particlesZ,
                z.data(),
                sizeof (float) * nParticles,
                cudaMemcpyHostToDevice,
                pst->lcl->streams(0)
        ));
    }

    return sizeof(output);
}

int ServiceCopyParticles::Combine(void *vout,void *vout2,int nIn,int nOut1,int nOut2) {

    return 0;
}
