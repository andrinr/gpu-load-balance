#include "reshuffleService.h"
#include "../cell.h"
#include <blitz/array.h>
// Make sure that the communication structure is "trivial" so that it
// can be moved around with "memcpy" which is required for MDL.
static_assert(std::is_void<ServiceReshuffle::input>()  || std::is_trivial<ServiceReshuffle::input>());
static_assert(std::is_void<ServiceReshuffle::output>() || std::is_trivial<ServiceReshuffle::output>());


static void swap(blitz::Array<float, 2> p, int a, int b) {
    for (int d = 0; d < p.columns(); d++) {
        float tmp = p(a, d);
        p(a, d) = p(b, d);
        p(b, d) = tmp;
    }
};

int ServiceReshuffle::Service(PST pst,void *vin,int nIn,void *vout, int nOut) {

    auto lcl = pst->lcl;
    auto in  = static_cast<input *>(vin);
    auto nCells = nIn / sizeof(input);

    // todo: either make x-y-z swaps directly in loop or do it after and perform memcpy
    for (int cellPtrOffset = 0; cellPtrOffset < nCells; ++cellPtrOffset){
        auto cell = static_cast<Cell>(*(in + cellPtrOffset));

        int beginInd = pst->lcl->cellToRangeMap(cell.id, 0);
        int endInd = pst->lcl->cellToRangeMap(cell.id, 1);

        int i = beginInd-1, j = endInd;
        float cut = cell.getCut();

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

        swap(lcl->particles, i, endInd -1);

        //printf("shuffle index %i %i %i cell %i cut %f \n", lcl->cellToRangeMap(cell.id, 0), i, lcl->cellToRangeMap(cell.id, 1), cell.id, cut);
        // todo: check if true ,might be +- 1
        lcl->cellToRangeMap(cell.getLeftChildId(), 0) =
                lcl->cellToRangeMap(cell.id, 0);
        lcl->cellToRangeMap(cell.getLeftChildId(), 1) = i;

        lcl->cellToRangeMap(cell.getRightChildId(), 0) = i;
        lcl->cellToRangeMap(cell.getRightChildId(), 1) =
                lcl->cellToRangeMap(cell.id, 1);

    }

    return 0;
}

int ServiceReshuffle::Combine(void *vout,void *vout2,int nIn,int nOut1,int nOut2) {
    return 0;
}
