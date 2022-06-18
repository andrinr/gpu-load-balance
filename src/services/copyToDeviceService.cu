#include "copyToDeviceService.h"
#include <blitz/array.h>
#include <vector>

// Make sure that the communication structure is "trivial" so that it
// can be moved around with "memcpy" which is required for MDL.
static_assert(std::is_void<ServiceCopyToDevice::input>()  || std::is_trivial<ServiceCopyToDevice::input>());
static_assert(std::is_void<ServiceCopyToDevice::output>() || std::is_trivial<ServiceCopyToDevice::output>());

int ServiceCopyToDevice::Service(PST pst,void *vin,int nIn,void *vout, int nOut) {
    // store streams / initialize in local data
    //
    auto lcl = pst->lcl;
    auto in  = static_cast<input *>(vin);
    auto out = static_cast<output *>(vout);
    auto nCells = nIn / sizeof(input);
    assert(nOut / sizeof(output) >= nCells);
    //printf("ServiceCopyToDevice invoked on thread %d\n",pst->idSelf);

    const int nThreads = 512;
    // Can increase speed by another factor of around two
    const int elementsPerThread = 16;

    auto cell = static_cast<Cell>(*(in + cellPtrOffset));

    int nParticles = lcl->particles.rows();

    const int nBlocks = ceil(nParticles / (nThreads * elementsPerThread) / 2 );

    //cudaStreamSynchronize(lcl->streams(streamId));
    int * d_counts;
    pst->lcl->d_counts(cellPtrOffset) = d_counts;

    blitz::Array<float,1> particles = pst->lcl->particles(blitz::Range(beginInd, endInd), 0);

    cudaMalloc(&d_particles, sizeof (float) * nParticles);
    cudaMalloc(&lcl->d_counts(cellPtrOffset), sizeof (int) * nBlocks);
    cudaMemcpyAsync(
            d_particles,
            particles.data(),
            endInd - beginInd,
            cudaMemcpyHostToDevice,
            pst->lcl->streams(streamId)
    );


    return nCells * sizeof(output);
}

int ServiceCopyToDevice::Combine(void *vout,void *vout2,int nIn,int nOut1,int nOut2) {

    return 0;
}
