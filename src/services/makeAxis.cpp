#include "makeAxis.h"
#include "../cell.h"
#include <blitz/array.h>
// Make sure that the communication structure is "trivial" so that it
// can be moved around with "memcpy" which is required for MDL.
static_assert(std::is_void<ServiceAxisSwap::input>()  || std::is_trivial<ServiceAxisSwap::input>());
static_assert(std::is_void<ServiceAxisSwap::output>() || std::is_trivial<ServiceAxisSwap::output>());

int ServiceAxisSwap::Service(PST pst,void *vin,int nIn,void *vout, int nOut) {

    auto lcl = pst->lcl;
    auto in  = static_cast<input *>(vin);
    auto out = static_cast<output *>(vout);
    auto nCells = nIn / sizeof(input);

    for (int cellPtrOffset = 0; cellPtrOffset < nCells; ++cellPtrOffset){
        auto cell = static_cast<Cell>(*(in + cellPtrOffset));

        int beginInd = pst->lcl->cellToRangeMap(cell.id, 0);
        int endInd = pst->lcl->cellToRangeMap(cell.id, 1);

        blitz::Array<float, 1> A =
                lcl->particles(blitz::Range(beginInd, endInd), cell.cutAxis);

        blitz::Array<float, 1> target =
                lcl->particlesAxis(blitz::Range(beginInd, endInd));

        std::memcpy(target.data(), A.data(), sizeof (float ) *  A.rows());

    }

    return sizeof (output);
}

int ServiceAxisSwap::Combine(void *vout,void *vout2,int nIn,int nOut1,int nOut2) {
    return 0;
}
