#include "countService.h"

// Make sure that the communication structure is "trivial" so that it
// can be moved around with "memcpy" which is required for MDL.
static_assert(std::is_void<ServiceCountLeft::input>()  || std::is_trivial<ServiceCountLeft::input>());
static_assert(std::is_void<ServiceCountLeft::output>() || std::is_trivial<ServiceCountLeft::output>());

int ServiceCount::Service(PST pst,void *vin,int nIn,void *vout, int nOut) {
    auto lcl = pst->lcl;
    auto in  = static_cast<input *>(vin);
    auto out = static_cast<output *>(vout);
    auto nCells = nIn / sizeof(input);

    vout[0] = lcl->cellToRangeMap(in[nCells - 1].id, 1) - lcl->cellToRangeMap(in[0].id, 0);

    return sizeof(output);
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
