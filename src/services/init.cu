#include "init.h"
#include <blitz/array.h>
#include <limits>
#include "../constants.h"
#include "../tipsy/tipsy.h"
// Make sure that the communication structure is "trivial" so that it
// can be moved around with "memcpy" which is required for MDL.
static_assert(std::is_void<ServiceInit::input>()  || std::is_trivial<ServiceInit::input>());
static_assert(std::is_void<ServiceInit::output>() || std::is_trivial<ServiceInit::output>());

static unsigned long x=123456789, y=362436069, z=521288629;

float xorshf96(void) {//period 2^96-1
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
    auto lcl = pst->lcl;
    ServiceInit::input in = *static_cast<input *>(vin);

    // Init positions
    blitz::GeneralArrayStorage<2> storage;
    storage.ordering() = 0,1;
    storage.base() = 0, 0;
    storage.ascendingFlag() = true, true;
    // x, y, z, cellId, tmp
    int k = 3;
    float * particlesXYZData = (float *)malloc(in.nParticles * sizeof(float ) * k);
    CUDA_CHECK(cudaMallocHost, ((void**)&particlesXYZData, in.nParticles * sizeof (float ) * k));

    auto particles = blitz::Array<float, 2>(
            particlesXYZData,
            blitz::shape(in.nParticles, k),
            blitz::deleteDataWhenDone,
            storage);

    if (in.generate) {
        for (int i = 0; i < in.nParticles; i++) {
            for (int d = 0; d < 3; d++) {
                particles(i,d) = xorshf96();
            }
        }
    }
    else {
        TipsyIO io;
        io.open("../tipsy/b0-final.std");
        printf("Count %d n %d\n", io.count(), in.nParticles);
        io.load(particles);
    }
    lcl->particles.reference(particles);

    // Mapping from cellId to particle index
    auto cellToRangeMap = blitz::Array<unsigned int, 2>(MAX_CELLS, 2);
    cellToRangeMap(0, 0) = 0;
    cellToRangeMap(0, 1) = in.nParticles;
    lcl->cellToRangeMap.reference(cellToRangeMap);

    // Temporary array buffer on the CPU
    if (not in.params.GPU_COUNT) {
        auto particlesT = blitz::Array<float, 1>(in.nParticles);
        lcl->particlesT.reference(particlesT);
    }
    else if (in.params.GPU_COUNT and not in.params.GPU_PARTITION) {
        float * particlesTData = (float *)malloc(in.nParticles * sizeof(float ));
        CUDA_CHECK(cudaMallocHost, ((void**)&particlesTData, in.nParticles * sizeof (float )));
        auto particlesT = blitz::Array<float, 1>(
                particlesTData,
                in.nParticles,
                blitz::deleteDataWhenDone);

        lcl->particlesT.reference(particlesT);
    }


    if (in.params.GPU_COUNT) {;

        int nBlocks;

        if (not in.params.GPU_PARTITION) {
            nBlocks = (int) ceil((float) in.nParticles / (N_THREADS * ELEMENTS_PER_THREAD)) + MAX_CELLS;
        } else {
            nBlocks = (int) ceil((float) in.nParticles / (N_THREADS * 2)) + MAX_CELLS;

            CUDA_CHECK(cudaMalloc,(&lcl->d_permutations, sizeof (unsigned int) * in.nParticles));
            CUDA_CHECK(cudaMalloc,(&lcl->d_particlesX, sizeof (float ) * in.nParticles));
            CUDA_CHECK(cudaMalloc,(&lcl->d_particlesY, sizeof (float ) * in.nParticles));
            CUDA_CHECK(cudaMalloc,(&lcl->d_particlesZ, sizeof (float ) * in.nParticles));

            lcl->h_cellBegin = (unsigned int*)malloc(nBlocks * sizeof(unsigned int));
            CUDA_CHECK(cudaMalloc,(&lcl->d_cellBegin, sizeof (unsigned int) * nBlocks));
            lcl->h_nLefts = (unsigned int*)malloc(nBlocks * sizeof(unsigned int));
            CUDA_CHECK(cudaMalloc,(&lcl->d_nLefts, sizeof (unsigned int) * nBlocks));
            lcl->h_cellIndices = (unsigned int*)malloc(nBlocks * sizeof(unsigned int));
            CUDA_CHECK(cudaMalloc,(&lcl->d_cellIndices, sizeof (unsigned int) * nBlocks));
            lcl->h_axis = (unsigned int*)malloc(nBlocks * sizeof(unsigned int));
            CUDA_CHECK(cudaMalloc,(&lcl->d_axis, sizeof (unsigned int ) * nBlocks));

            auto h_countsLeft = blitz::Array<unsigned int, 1>(MAX_CELLS);
            lcl->h_countsLeft.reference(h_countsLeft);

            CUDA_CHECK(cudaMalloc,(&lcl->d_offsetLeq, sizeof (unsigned int) * MAX_CELLS));
            CUDA_CHECK(cudaMalloc,(&lcl->d_offsetG, sizeof (unsigned int) * MAX_CELLS));
        }

        CUDA_CHECK(cudaMalloc, (&lcl->d_begins, sizeof (unsigned int) * nBlocks));
        CUDA_CHECK(cudaMalloc, (&lcl->d_ends, sizeof (unsigned int) * nBlocks));
        CUDA_CHECK(cudaMalloc, (&lcl->d_cuts, sizeof (float) * nBlocks));

        lcl->h_begins = (unsigned int*)malloc(nBlocks * sizeof(unsigned int));
        CUDA_CHECK(cudaMallocHost, ((void**)&lcl->h_begins, nBlocks * sizeof (unsigned int)));
        lcl->h_ends = (unsigned int*)malloc(nBlocks * sizeof(unsigned int));
        CUDA_CHECK(cudaMallocHost, ((void**)&lcl->h_ends, nBlocks * sizeof (unsigned int)));
        lcl->h_cuts = (float*)malloc(nBlocks * sizeof(float));
        CUDA_CHECK(cudaMallocHost, ((void**)&lcl->h_cuts, nBlocks * sizeof (float)));

        CUDA_CHECK(cudaMalloc, (&lcl->d_results, sizeof (unsigned int) * nBlocks));
        lcl->h_results = (unsigned int*)malloc(nBlocks * sizeof(unsigned int));
        CUDA_CHECK(cudaMallocHost, ((void**)&lcl->h_results, nBlocks * sizeof (unsigned int)));

        // Temporary particle array on the GPU
        CUDA_CHECK(cudaMalloc,(&lcl->d_particlesT, sizeof (float) * in.nParticles));

        // Streams
        auto streams = blitz::Array<cudaStream_t , 1>(N_STREAMS);
        for (int i = 0; i < N_STREAMS; i++) {
            cudaStream_t stream;
            CUDA_CHECK(cudaStreamCreate, (&stream));
            streams(i) = stream;
        }
        lcl->streams.reference(streams);
    }

    return 0;
}

int ServiceInit::Combine(void *vout,void *vout2,int nIn,int nOut1,int nOut2) {
    return 0;
}
