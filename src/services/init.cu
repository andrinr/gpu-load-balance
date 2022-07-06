#include "init.h"
#include <blitz/array.h>
#include <limits>
#include "../constants.h"
#include "../data/tipsy.h"

// Make sure that the communication structure is "trivial" so that it
// can be moved around with "memcpy" which is required for MDL.
static_assert(std::is_void<ServiceInit::input>()  || std::is_trivial<ServiceInit::input>());
static_assert(std::is_void<ServiceInit::output>() || std::is_trivial<ServiceInit::output>());

static unsigned long x=123456789, y=362436069, z=521288629;

float xorshf96(void) {          //period 2^96-1
    unsigned long t;
    x ^= x << 16;
    x ^= x >> 5;
    x ^= x << 1;

    t = x;
    x = y;
    y = z;
    z = t ^ x ^ y;

    return (float) z / std::numeric_limits<unsigned long>::max() - 0.5;
}

int ServiceInit::Service(PST pst,void *vin,int nIn,void *vout, int nOut) {

    printf("ServiceInit invoked on thread %d\n",pst->idSelf);

    auto lcl = pst->lcl;
    ServiceInit::input in = *static_cast<input *>(vin);

    // Init positions
    blitz::GeneralArrayStorage<2> storage;
    storage.ordering() = 0,1;
    storage.base() = 0, 0;
    storage.ascendingFlag() = true, true;
    // x, y, z, cellId, tmp
    int k = 3;
    auto particles = blitz::Array<float, 2>(in.nParticles, k, storage);

    float * particlesAxisData = (float *)calloc(N, sizeof(float ));
    CUDA_CHECK(cudaMallocHost, ((void**)&particlesAxisData, N * sizeof (float )));

    auto particlesAxis = blitz::Array<float, 1>(
            particlesAxisData,
            in.nParticles,
            blitz::deleteDataWhenDone);
    auto cellToRangeMap = blitz::Array<uint, 2>(MAX_CELLS, 2);
    float * d_particles;

    //auto tipsy = TipsyIO("../data/tipsy/b0-final.std");
    //tipsy.load()

    srand(pst->idSelf);
    int c = 0;
    for (int i = 0; i < in.nParticles; i++) {
        for (int d = 0; d < 3; d++) {
            particles(i,d) = xorshf96();
            if (particles(i,d) < 0.0) c++;
        }
    }

    printf("ServiceInit generated random numbers %d\n",pst->idSelf);

    cellToRangeMap(0, 0) = 0;
    cellToRangeMap(0, 1) = in.nParticles;

    const int nBlocks = (int) ceil((float) in.nParticles / (N_THREADS * ELEMENTS_PER_THREAD)) + MAX_CELLS;
    unsigned int* h_results = (unsigned int*)malloc(nBlocks * sizeof(unsigned int));
    CUDA_CHECK(cudaMallocHost, ((void**)&h_results, nBlocks * sizeof (unsigned int)));

    lcl->h_results = h_results;
    lcl->particles.reference(particles);
    lcl->particlesAxis.reference(particlesAxis);
    lcl->d_particles = d_particles;
    lcl->cellToRangeMap.reference(cellToRangeMap);

    CUDA_CHECK(cudaMalloc,(&lcl->d_particles, sizeof (float) * in.nParticles));

    CUDA_CHECK(cudaMalloc, (&lcl->d_results, sizeof (unsigned int) * nBlocks));

    auto streams = blitz::Array<cudaStream_t , 1>(N_STREAMS);

    for (int i = 0; i < N_STREAMS; i++) {
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate, (&stream));
        streams(i) = stream;
    }

    lcl->streams.reference(streams);

    printf("ServiceInit finished on thread %d\n",pst->idSelf);

    return 0;
}

int ServiceInit::Combine(void *vout,void *vout2,int nIn,int nOut1,int nOut2) {
    return 0;
}
