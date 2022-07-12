#include "count.h"

// Make sure that the communication structure is "trivial" so that it
// can be moved around with "memcpy" which is required for MDL.
static_assert(std::is_void<ServiceCount::input>()  || std::is_trivial<ServiceCount::input>());
static_assert(std::is_void<ServiceCount::output>() || std::is_trivial<ServiceCount::output>());

int ServiceCount::Service(PST pst,void *vin,int nIn,void *vout, int nOut) {
    auto lcl = pst->lcl;
    auto in  = static_cast<input *>(vin);
    auto out = static_cast<output *>(vout);
    auto nCells = nIn / sizeof(input);

    for (int cellPtrOffset = 0; cellPtrOffset < nCells; ++cellPtrOffset) {
        auto cell = static_cast<Cell>(*(in + cellPtrOffset));
        out[cellPtrOffset] = lcl->cellToRangeMap(cell.id,1) - lcl->cellToRangeMap(cell.id,0);

    }
    return nCells * sizeof (output);
}

int ServiceCount::Combine(void *vout,void *vout2,int nIn,int nOut1,int nOut2) {
    auto out  = static_cast<output *>(vout);
    auto out2 = static_cast<output *>(vout2);
    int nCounts = nIn / sizeof(input);
    assert(nOut1 >= nCounts*sizeof(output));
    assert(nOut2 >= nCounts*sizeof(output));
    for(auto i=0; i<nCounts; ++i)
	    out[i] += out2[i];
    return nCounts * sizeof(output);
}
