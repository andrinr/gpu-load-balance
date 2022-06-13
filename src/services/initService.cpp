#include "initService.h"
#include <blitz/array.h>

// Make sure that the communication structure is "trivial" so that it
// can be moved around with "memcpy" which is required for MDL.
static_assert(std::is_void<ServiceInit::input>()  || std::is_trivial<ServiceInit::input>());
static_assert(std::is_void<ServiceInit::output>() || std::is_trivial<ServiceInit::output>());

int ServiceInit::Service(PST pst,void *vin,int nIn,void *vout, int nOut) {

    printf("ServiceInit invoked on thread %d\n",pst->idSelf);

    // Init positions
    blitz::GeneralArrayStorage<2> storage;
    storage.ordering() = 0,1;
    storage.base() = 0, 0;
    storage.ascendingFlag() = true, true;

    int nStreams = 32;
    int n = 1 << 16;
    int k = 4;

    auto particles = blitz::Array<float, 2>(n, k, storage);
    auto cellToRangeMap = blitz::Array<int, 2>(max_cells, 2);
    auto streams = blitz::Array<cudaStream_t, 1>(nStreams);
    auto d_particles = blitz::Array<float *, 1>(max_cells);
    auto d_counts = blitz::Array<int *, 1>(max_cells);

    srand(pst->idSelf);
    int c = 0;
    for (int i = 0; i < n; i++) {
        for (int d = 0; d < 3; d++) {
            particles(i,d) = (float)(rand())/(float)(RAND_MAX) - 0.5;
            if (particles(i,d) < 0.0) c++;
        }
    }

    printf("ServiceInit generated random numbers %d\n",pst->idSelf);

    for (int i = 0; i < nStreams; i++) {
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        streams(i) = stream;
    }

    cellToRangeMap(0, 0) = 0;
    cellToRangeMap(0, 1) = n;

    auto lcl = pst->lcl;
    lcl->particles.reference(particles);
    lcl->d_particles.reference(d_particles);
    lcl->cellToRangeMap.reference(cellToRangeMap);
    lcl->streams.reference(streams);
    lcl->d_counts.reference(d_counts);

    //pst->lcl = new LocalData(particles, cellToRangeMap, streams, d_particles, d_counts);

    printf("ServiceInit finished on thread %d\n",pst->idSelf);

    return 0;
}

int ServiceInit::Combine(void *vout,void *vout2,int nIn,int nOut1,int nOut2) {
    return 0;
}
