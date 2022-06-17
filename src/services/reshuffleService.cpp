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

    // todo: either make x-y-z swaps directly in loop or do it after and perform memcpy
    for (int cellPtrOffset = 0; cellPtrOffset < nCells; ++cellPtrOffset){
        auto cell = static_cast<Cell>(*(in + cellPtrOffset));

        int beginInd = pst->lcl->cellToRangeMap(cell.id, 0);
        int endInd = pst->lcl->cellToRangeMap(cell.id, 1);

        int i = beginInd-1, j = endInd;
        float cut =  (cell.cutMarginLeft + cell.cutMarginRight) / 2.0;

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

        // todo: check if true ,might be +- 1
        swap(lcl->particles, i, endInd -1);

        lcl->cellToRangeMap(cell.getLeftChildId(), 0) =
                lcl->cellToRangeMap(cell.id, 0);
        lcl->cellToRangeMap(cell.getLeftChildId(), 1) = i;

        lcl->cellToRangeMap(cell.getRightChildId(), 0) = i;
        lcl->cellToRangeMap(cell.getRightChildId(), 1) =
                lcl->cellToRangeMap(cell.id, 1);

        out[cellPtrOffset] = i;

        if (cell.prevCutAxis == cell.cutAxis) continue;

        // Restore default
        if (cell.prevCutAxis != 0) {
            blitz::Array<float, 1> a =
                    lcl->particles(blitz::Range(beginInd, endInd), cell.prevCutAxis);
            blitz::Array<float, 1> b =
                    lcl->particles(blitz::Range(beginInd, endInd), 0);
            blitz::Array<float, 1> tmp =
                    lcl->particles(blitz::Range(beginInd, endInd), 4);

            std::memcpy(&tmp.data(), &a.data(), sizeof (float ) *  a.rows());
            std::memcpy(&a.data(), &b.data(), sizeof (float ) *  a.rows());
            std::memcpy(&b.data(), &tmp.data(), sizeof (float ) *  a.rows());
        }

        if (cell.cutAxis != 0) {
            blitz::Array<float, 1> a =
                    lcl->particles(blitz::Range(beginInd, endInd), cell.cutAxis);
            blitz::Array<float, 1> b =
                    lcl->particles(blitz::Range(beginInd, endInd), 0);
            blitz::Array<float, 1> tmp =
                    lcl->particles(blitz::Range(beginInd, endInd), 4);

            std::memcpy(&tmp.data(), &a.data(), sizeof (float ) *  a.rows());
            std::memcpy(&a.data(), &b.data(), sizeof (float ) *  a.rows());
            std::memcpy(&b.data(), &tmp.data(), sizeof (float ) *  a.rows());
        }
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
