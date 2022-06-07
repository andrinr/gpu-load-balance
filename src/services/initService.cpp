#include "initService.h"
#include <blitz/array.h>

// Make sure that the communication structure is "trivial" so that it
// can be moved around with "memcpy" which is required for MDL.
static_assert(std::is_void<ServiceCountLeft::input>()  || std::is_trivial<ServiceCountLeft::input>());
static_assert(std::is_void<ServiceCountLeft::output>() || std::is_trivial<ServiceCountLeft::output>());

int ServiceInitParticles::Service(PST pst,void *vin,int nIn,void *vout, int nOut) {

    cudaError_t result;

    int n = 1 << 20;
    int k = 4;
    blitz::Array<float, 2> p = blitz::Array<float, 2>(n, k);
    blitz::Array<int, 2> CTRM = blitz::Array<float, 2>(max_cells, 2);
    blitz::Array<cudaStream_t, 1> cudaStreams = blitz::Array<cudaStream_t, 1>(max_cells / 2.0);
    blitz::Array<float *, 1> d_particlesPtrs + blitz::Array<float *, 1>(max_cells / 2.0 / 2.0);

    p = 0;
    CTRM = -1;
    // srand(vmdl);
    for (int i = 0; i < n; i++) {
        for (int d = 0; d < 3; d++) {
            p(i,d) = (float)(rand())/(float)(RAND_MAX);
        }
        cudaStream_t stream;
        result = cudaStreamCreate(&stream);
        cudaStreams(i) = stream;
    }

    CTRM(0, 0) = 0;
    CTRM(0, 1) = n - 1;

    pst->lcl = new LocalData();

    pst->lcl->particles = p;
    pst->lcl->cellToRangeMap = CTRM;
    pst->lcl->streams = cudaStreams;

}

int ServiceInitParticles::Combine(void *vout,void *vout2,int nIn,int nOut1,int nOut2) {
    return 0;
}
