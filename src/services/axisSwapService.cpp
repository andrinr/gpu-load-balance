#include "axisSwapService.h"
#include "../cell.h"
#include <blitz/array.h>
// Make sure that the communication structure is "trivial" so that it
// can be moved around with "memcpy" which is required for MDL.
static_assert(std::is_void<ServiceAxisSwap::input>()  || std::is_trivial<ServiceAxisSwap::input>());
static_assert(std::is_void<ServiceAxisSwap::output>() || std::is_trivial<ServiceAxisSwap::output>());

static void swap(blitz::Array<float, 2> & p, int a, int begin, int end) {
    blitz::Array<float, 1> A =
            p(blitz::Range(begin, end), a);
    blitz::Array<float, 1> B =
            p(blitz::Range(begin, end), 0);
    blitz::Array<float, 1> tmp =
            p(blitz::Range(begin, end), 3);

    std::memcpy(tmp.data(), A.data(), sizeof (float ) *  A.rows());
    std::memcpy(A.data(), B.data(), sizeof (float ) *  A.rows());
    std::memcpy(B.data(), tmp.data(), sizeof (float ) *  A.rows());
}

int ServiceAxisSwap::Service(PST pst,void *vin,int nIn,void *vout, int nOut) {

    auto lcl = pst->lcl;
    auto in  = static_cast<input *>(vin);
    auto out = static_cast<output *>(vout);
    auto nCells = nIn / sizeof(input);

    // todo: either make x-y-z swaps directly in loop or do it after and perform memcpy
    for (int cellPtrOffset = 0; cellPtrOffset < nCells; ++cellPtrOffset){
        auto cell = static_cast<Cell>(*(in + cellPtrOffset));

        int beginInd = pst->lcl->cellToRangeMap(cell.id, 0);
        int endInd = pst->lcl->cellToRangeMap(cell.id, 1);

        if (cell.prevCutAxis == cell.cutAxis) continue;

        // Restore default
        if (cell.prevCutAxis > 0) {
            swap(lcl->particles, cell.prevCutAxis, beginInd, endInd);
            //printf("cell %u, swap 0 and %u \n", cell.id, cell.prevCutAxis);
        }

        // Rearrange axis layout
        if (cell.cutAxis > 0) {
            swap(lcl->particles, cell.cutAxis, beginInd, endInd);
            //printf("cell %u, swap %u and 0 \n", cell.id, cell.cutAxis);
        }
    }

    return sizeof (output);
}

int ServiceAxisSwap::Combine(void *vout,void *vout2,int nIn,int nOut1,int nOut2) {
    return 0;
}
