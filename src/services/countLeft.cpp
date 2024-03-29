#include "countLeft.h"
#include "../cell.h"
// Make sure that the communication structure is "trivial" so that it
// can be moved around with "memcpy" which is required for MDL.
static_assert(std::is_void<ServiceCountLeft::input>()  || std::is_trivial<ServiceCountLeft::input>());
static_assert(std::is_void<ServiceCountLeft::output>() || std::is_trivial<ServiceCountLeft::output>());


int ServiceCountLeft::Service(PST pst, void *vin, int nIn, void *vout, int nOut) {
    auto lcl = pst->lcl;
    auto in  = static_cast<input *>(vin);
    auto out = static_cast<output *>(vout);
    auto nCells = nIn / sizeof(input);
    assert(nOut / sizeof(output) >= nCells);

    for (int cellPtrOffset = 0; cellPtrOffset < nCells; ++cellPtrOffset){
        auto cell = static_cast<Cell>(*(in + cellPtrOffset));

        if (cell.foundCut) {
            continue;
        }
        int beginInd = pst->lcl->cellToRangeMap(cell.id, 0);
        int endInd =  pst->lcl->cellToRangeMap(cell.id, 1);

        blitz::Array<float,1> particles =
                pst->lcl->particlesT(blitz::Range(beginInd, endInd));

        float * startPtr = particles.data();
        float * endPtr = startPtr + (endInd - beginInd);

        int nLeft = 0;
        float cut = cell.getCut();
        for(auto p= startPtr; p<endPtr; ++p)
        {
            nLeft += *p < cut;
        }

        out[cellPtrOffset] = nLeft;
    }

    return nCells * sizeof(output);
}

int ServiceCountLeft::Combine(void *vout,void *vout2,int nIn,int nOut1,int nOut2) {
    auto out  = static_cast<output *>(vout);
    auto out2 = static_cast<output *>(vout2);
    int nCounts = nIn / sizeof(input);
    assert(nOut1 >= nCounts*sizeof(output));
    assert(nOut2 >= nCounts*sizeof(output));
    for(auto i=0; i<nCounts; ++i)
	    out[i] += out2[i];
    return nCounts * sizeof(output);
}
