#include <tasks.h>
#include "comm/comm.h"
#include <math.h>
#include <blitz/array.h>
#include "services.h"

void Tasks::operate(
        Orb& orb,
        Comm comm,
        blitz::Array<Cell, 1> cells,
        int nLeafCells
    ){
    // loop over levels of tree
    for (int l = 1; l < ceil(log2(nLeafCells)); l++) {
        int begin_prev = 2**(l-1);
        int end_prev = 2**l;
        int begin = end_prev;
        int end = 2**(l+1);

        // Init cells
        for (int i = begin; i < end; i++) {
            float maxValue = 0;
            int axis = -1;

            for (int i = 0; i < DIMENSIONS; i++) {
                float size = cells(i).upper[i] - cells(i).lower[i];
                if (size > maxValue) {
                    maxValue = size;
                    axis = i;
                }
            }

            cells(i).cutAxis = axis;
            cells(i).left = lower[axis];
            cells(i).right = upper[axis];
        }

        // Loop
        bool foundAll = false;
        while(!foundAll) {
            InCount inCount(
                cells(blitz::Range(begin, end)),
                orb,
                begin,
                end);

            comm.dispatchService(Services::count, )
            comm.dispatchWork(&begin, cells(blitz::Range(begin, end)));
            work(&orb, &comm);
            comm.concludeWork(&begin, )

            for (int i = begin; i < min(nLeafCells, end); i++) {

                if


                cells(i).left = lower[axis];
                cells(i).right = upper[axis];
            }



        }

        int mid = reshuffleArray(axis, begin, end, cut);


        float cut = findCut(cell, axis, begin, end);
        int mid = reshuffleArray(axis, begin, end, cut);
        int nCellsLeft = ceil(cell.nCells / 2.0);
        int nCellsRight = cell.nCells - nCellsLeft;
        Cell leftChild (cells.size(), -1, nCellsLeft, cell.lower, cell.upper);

        cellBegin.push_back(begin);
        cellEnd.push_back(mid);
        assign(begin, mid, cells.size());

        leftChild.upper[axis] = cut;

        if (nCellsLeft > 1) {
            stack.push(cells.size());
        }
        cells.push_back(leftChild);

        Cell rightChild (cells.size(), -1, nCellsRight, cell.lower, cell.upper);

        cellBegin.push_back(mid);
        cellEnd.push_back(end);
        assign(mid, end, cells.size());

        rightChild.lower[axis] = cut;

        if (nCellsRight > 1) {
            stack.push(cells.size());
        }

        cells.push_back(rightChild);
    }

    
}

void Tasks::work(Orb& orb, MPI_Comm comm) {
    int n;
    Cell cells[nLeafCells * 2];

    while(true) {
        MPI_Comm::dispatchWork(&n, &cells);

        // Check weather still work to do
        if (n == -1) {
            break;
        }

        int* count = orb.count();

        MPI_Comm::concludeWork(n, count);
    }
}

