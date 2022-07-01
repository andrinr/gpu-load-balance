#include "init.cuh"
#include <blitz/array.h>
#include "../constants.h"
// Make sure that the communication structure is "trivial" so that it
// can be moved around with "memcpy" which is required for MDL.
static_assert(std::is_void<ServiceInitGPU::input>()  || std::is_trivial<ServiceInitGPU::input>());
static_assert(std::is_void<ServiceInitGPU::output>() || std::is_trivial<ServiceInitGPU::output>());

int ServiceInitGPU::Service(PST pst,void *vin,int nIn,void *vout, int nOut) {

    auto lcl = pst->lcl;
    ServiceInitGPU::input in = *static_cast<input *>(vin);

    CUDA_CHECK(cudaMalloc,(&lcl->d_particles, sizeof (float) * in.nParticles));

    int nCounts = (in.nParticles / (N_THREADS * 2 * ELEMENTS_PER_THREAD) + MAX_CELLS);
    printf("nCounts %i \n", nCounts);
    CUDA_CHECK(cudaMalloc, (&lcl->d_resultsA, sizeof (uint) * nCounts));
    CUDA_CHECK(cudaMalloc, (&lcl->d_resultsB, sizeof (uint) * nCounts));

    auto streams = blitz::Array<cudaStream_t , 1>(N_STREAMS);

    for (int i = 0; i < N_STREAMS; i++) {
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate, (&stream));
        streams(i) = stream;
    }

    lcl->streams.reference(streams);

    return 0;
}

int ServiceInitGPU::Combine(void *vout,void *vout2,int nIn,int nOut1,int nOut2) {
    return 0;
}
