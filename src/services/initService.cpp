#include "initService.h"
#include <blitz/array.h>

// Make sure that the communication structure is "trivial" so that it
// can be moved around with "memcpy" which is required for MDL.
static_assert(std::is_void<ServiceInit::input>()  || std::is_trivial<ServiceInit::input>());
static_assert(std::is_void<ServiceInit::output>() || std::is_trivial<ServiceInit::output>());

int ServiceInit::Service(PST pst,void *vin,int nIn,void *vout, int nOut) {

    printf("ServiceInit invoked on thread %d\n",pst->idSelf);

    auto lcl = pst->lcl;
    auto in  = static_cast<input *>(vin);
    auto out = static_cast<output *>(vout);
    auto nCells = nIn / sizeof(input);

    cudaError_t result;

    int n = 1 << 10;
    int k = 4;
    blitz::Array<float, 2> p = blitz::Array<float, 2>(n, k);
    blitz::Array<int, 2> CTRM = blitz::Array<int, 2>(max_cells, 2);
    blitz::Array<cudaStream_t, 1> cudaStreams = blitz::Array<cudaStream_t, 1>(max_cells);
    blitz::Array<float *, 1> d_particlesPtrs = blitz::Array<float *, 1>(max_cells);

    p = 0;
    CTRM = -1;
    // srand(vmdl);
    for (int i = 0; i < n; i++) {
        for (int d = 0; d < 3; d++) {
            p(i,d) = (float)(rand())/(float)(RAND_MAX);
        }
    }

    printf("ServiceInit generated random numbers %d\n",pst->idSelf);


    for (int i = 0; i < 32; i++) {
        cudaStream_t stream;
        result = cudaStreamCreate(&stream);
        cudaStreams(i) = stream;
    }

    printf("ServiceInit generated streams %d\n",pst->idSelf);

    CTRM(0, 0) = 0;
    CTRM(0, 1) = n - 1;

    pst->lcl = new LocalData();

    pst->lcl->particles = p;
    pst->lcl->cellToRangeMap = CTRM;
    pst->lcl->streams = cudaStreams;

    printf("ServiceInit finished on thread %d\n",pst->idSelf);


    return 0;
}

int ServiceInit::Combine(void *vout,void *vout2,int nIn,int nOut1,int nOut2) {
    return 0;
}
