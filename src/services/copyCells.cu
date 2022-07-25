#include "copyCells.h"
#include <blitz/array.h>
#include <vector>
#include "../constants.h"
// Make sure that the communication structure is "trivial" so that it
// can be moved around with "memcpy" which is required for MDL.
static_assert(std::is_void<ServiceCopyCells::input>()  || std::is_trivial<ServiceCopyCells::input>());
static_assert(std::is_void<ServiceCopyCells::output>() || std::is_trivial<ServiceCopyCells::output>());

int ServiceCopyCells::Service(PST pst,void *vin,int nIn,void *vout, int nOut) {
    // store streams / initialize in local data
    auto lcl = pst->lcl;
    auto in  = static_cast<input *>(vin);
    unsigned int nCells = nIn / sizeof(input);

    int nParticles = lcl->particles.rows();
    // We only need the first nParticles, since axis 0 is axis where cuts need to be found

    int nBlocks = 0;
    int blockPtr = 0;

    printf("nCells: %d\n", nCells);

    for (int cellPtrOffset = 0; cellPtrOffset < nCells; ++cellPtrOffset) {

        auto cell = static_cast<Cell>(*(in + cellPtrOffset));
        unsigned int beginInd = lcl->cellToRangeMap(cell.id, 0);
        unsigned int endInd =  lcl->cellToRangeMap(cell.id, 1);
        unsigned int n = endInd - beginInd;

        unsigned int nBlocksPerCell = (int) ceil((float) n / (N_THREADS * ELEMENTS_PER_THREAD));

        int begin = beginInd;
        for (int i = 0; i < nBlocksPerCell; ++i) {
            lcl->h_begins[blockPtr] = begin;
            begin += N_THREADS * ELEMENTS_PER_THREAD;
            lcl->h_ends[blockPtr] = min(begin, endInd);
            blockPtr++;
        }
        nBlocks += nBlocksPerCell;

    }

    CUDA_CHECK(cudaMemcpyAsync,(
            lcl->d_begins,
            lcl->h_begins,
            sizeof (unsigned int) * nBlocks,
            cudaMemcpyHostToDevice,
            lcl->streams(0)
    ));

    CUDA_CHECK(cudaMemcpyAsync,(
            lcl->d_ends,
            lcl->h_ends,
            sizeof (unsigned int) * nBlocks,
            cudaMemcpyHostToDevice,
            lcl->streams(0)
    ));

    return 0;
}

int ServiceCopyCells::Combine(void *vout,void *vout2,int nIn,int nOut1,int nOut2) {

    return 0;
}
