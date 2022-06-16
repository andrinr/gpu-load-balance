#include "reshuffleService.h"
#include "../cell.h"
// Make sure that the communication structure is "trivial" so that it
// can be moved around with "memcpy" which is required for MDL.
static_assert(std::is_void<ServiceReshuffle::input>()  || std::is_trivial<ServiceReshuffle::input>());
static_assert(std::is_void<ServiceReshuffle::output>() || std::is_trivial<ServiceReshuffle::output>());

static void swap(blitz::Array<float, 2> &p, int a, int b) {
    for (int d = 0; d < p.columns(); d++) {
        float tmp = p(a, d);
        p(a, d) = p(b, d);
        p(b, d) = tmp;
    }
}

int ServiceReshuffle::Service(PST pst,void *vin,int nIn,void *vout, int nOut) {
    auto lcl = pst->lcl;
    auto in  = static_cast<input *>(vin);
    auto out = static_cast<output *>(vout);
    auto nCells = nIn / sizeof(input);
    for (int cellPtrOffset = 0; cellPtrOffset < nCells; ++cellPtrOffset){
        auto cell = static_cast<Cell>(*(in + cellPtrOffset));

        int beginInd = pst->lcl->cellToRangeMap(cell.id, 0);
        int endInd = pst->lcl->cellToRangeMap(cell.id, 1);

        int i = beginInd-1, j = endInd;
        float cut =  (cell.cutMarginLeft + cell.cutMarginRight) / 2.0;

        //printf("cut %f \n", cut);

        while(true)
        {
            do
            {
                i++;
            } while(lcl->particles(i, cell.cutAxis) < cut && i <= endInd);

            do
            {
                j--;
            } while(lcl->particles(j, cell.cutAxis) > cut && j >= beginInd);

            if(i >= j) {
                break;
            }

            swap(lcl->particles, i, j);
        }

        //printf("reshuffle %u, %u, %i, %i\n", beginInd, endInd, i, j);

        swap(lcl->particles, i, endInd -1);

        lcl->cellToRangeMap(CellHelpers::getLeftChildId(cell), 0) =
                lcl->cellToRangeMap(cell.id, 0);
        lcl->cellToRangeMap(CellHelpers::getLeftChildId(cell), 1) = i;

        lcl->cellToRangeMap(CellHelpers::getRightChildId(cell), 0) = i;
        lcl->cellToRangeMap(CellHelpers::getRightChildId(cell), 1) =
                lcl->cellToRangeMap(cell.id, 1);

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
