#include "reshuffleService.h"
#include "../cell.h"
// Make sure that the communication structure is "trivial" so that it
// can be moved around with "memcpy" which is required for MDL.
static_assert(std::is_void<ServiceReshuffle::input>()  || std::is_trivial<ServiceReshuffle::input>());
static_assert(std::is_void<ServiceReshuffle::output>() || std::is_trivial<ServiceReshuffle::output>());

int ServiceReshuffle::Service(PST pst,void *vin,int nIn,void *vout, int nOut) {
    auto lcl = pst->lcl;
    auto in  = static_cast<input *>(vin);
    auto out = static_cast<output *>(vout);
    auto nCells = nIn / sizeof(input);
    for (int cellPtrOffset = 0; cellPtrOffset < nCells; ++cellPtrOffset){
        auto cell = static_cast<Cell>(*(in + cellPtrOffset));

        int beginInd = pst->lcl->cellToRangeMap(cell.id, 0);
        int endInd = pst->lcl->cellToRangeMap(cell.id, 1);

        int i = beginInd;
        for (int j = beginInd; j < endInd; j++) {
            if (lcl->particles(j, cell.cutAxis) < (cell.cutMarginLeft + cell.cutMarginRight) / 2.0) {
                for (int d = 0; d < lcl->particles.columns(); d++) {
                    float tmp = lcl->particles(i, d);
                    lcl->particles(i, d) = lcl->particles(j, d);
                    lcl->particles(j, d) = tmp;
                }
                i = i + 1;
            }
        }

        lcl->cellToRangeMap(CellHelpers::getLeftChildId(cell), 1) = i;
        lcl->cellToRangeMap(CellHelpers::getRightChildId(cell), 0) = i;

        out[cellPtrOffset] = i;
    }
    return nCells * sizeof (output);
}

int ServiceReshuffle::Combine(void *vout,void *vout2,int nIn,int nOut1,int nOut2) {
    auto out  = static_cast<output *>(vout);
    auto out2 = static_cast<output *>(vout2);
    int nCounts = nIn / sizeof(input);
    assert(nOut1 >= nCounts*sizeof(output));
    assert(nOut2 >= nCounts*sizeof(output));
    for(auto i=0; i<nCounts; ++i)
	    out[i] += out2[i];
    return nCounts * sizeof(output);
}
