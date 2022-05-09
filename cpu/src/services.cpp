#include "services.h"
#include "comm/MPIMessaging.h"
#include <math.h>
#include <algorithm>

void Services::countLeft(Orb &orb, Cell *c, int *results, int n) {
    blitz::Array<int, 1> counts(n);
    for (int cellPtrOffset = 0; cellPtrOffset < n; cellPtrOffset++){
        Cell cell = *(c + cellPtrOffset);
        int beginInd = orb.cellToParticle(cell.id, 0);
        int endInd = orb.cellToParticle(cell.id, 1);

        blitz::Array<float,1> particles = orb.particles(blitz::Range(beginInd, endInd), cell.cutAxis);
        float * startPtr = particles.data();
        float * endPtr = startPtr + (endInd - beginInd);

        int nLeft = 0;

        float cut = (cell.cutMarginRight + cell.cutMarginLeft) / 2.0;
        for(auto p=startPtr; p<endPtr; ++p) nLeft += *p < cut;

        std::cout << nLeft << " ";
        results[cellPtrOffset] = nLeft;
    }
}

void Services::localReshuffle(Orb &orb, Cell *c, int *results, int n) {
    for (int cellPtrOffset = 0; cellPtrOffset < n; cellPtrOffset++){
        Cell cell = *(c + cellPtrOffset);
        int beginInd = orb.cellToParticle(cell.id, 0);
        int endInd = orb.cellToParticle(cell.id, 1);

        blitz::Array<float,1> particles = orb.particles(blitz::Range(beginInd, endInd), cell.cutAxis);
        float * startPtr = particles.data();
        float *endPtr = startPtr + (endInd - beginInd);

        int i = beginInd;
        for (int j = beginInd; j < endInd; j++) {
            if (orb.particles(j, cell.cutAxis) < (cell.cutMarginLeft + cell.cutMarginRight) / 2.0) {
                orb.swap(i, j);
                i = i + 1;
            }
        }

        orb.swap(i, endInd - 1);
        results[cellPtrOffset] = i;
    }
}

void Services::count(Orb &orb, Cell *c, int *results, int n) {

    std::cout << " counting " << n << std::endl;

    for (int i = 0; i < n; ++i) {
        Cell cell = c[i];
        cell.log();
        int begin = orb.cellToParticle(cell.id, 0);
        int end = orb.cellToParticle(cell.id, 1);

        results[i] = begin - end;
    }
}

void Services::buildTree(Orb &orb, Cell *c, int *results, int n) {

    // Blitz: 2.3.7
    blitz::Array<Cell, 1> cells(c, blitz::shape(n));

    Cell root = cells(0);

    // loop over levels of tree
    root.log();

    for (int l = 1; l < ceil(log2(root.nLeafCells)); l++) {

        int a = std::pow(2, (l-1)) - 1;
        int b = std::min((int)std::pow(2, l), root.nLeafCells) - 1;

        std::cout << l << "-level" << " a "  << a << " b " << b << std::endl;

        bool status;
        int* sum;
        std::tie(status, sum) = MPIMessaging::dispatchService(
                orb,
                countService,
                cells(blitz::Range(a, b-1)).data(),
                b - a,
                sum,
                b - a,
                std::make_tuple(1, MPIMessaging::np),
                0);

        std::cout << sum << "sum" << std::endl;


        blitz::Array<Cell, 1> countBlitz(sum, blitz::shape(b - a));
        // Loop
        bool foundAll = true;

        while(foundAll) {
            int* sumLeft;
            foundAll = true;
            std::tie(status, sumLeft) = MPIMessaging::dispatchService(
                    orb,
                    countLeftService,
                    cells(blitz::Range(a, b)).data(),
                    b - a,
                    sumLeft,
                    b - a,
                    std::make_tuple(1,MPIMessaging::np),
                    0);

            blitz::Array<int, 1> countLeftBlitz(sumLeft, blitz::shape(b - a));

            for (int i = a; i < b; i++) {

               if (abs(countLeftBlitz(i) - countLeftBlitz(i) / 2.0) < CUTOFF) {
                   cells(i).cutAxis = -1;
               }
               else if (countLeftBlitz(i) - countLeftBlitz(i) / 2.0 > 0){
                    cells(i).cutMarginRight = (cells(i).cutMarginLeft + cells(i).cutMarginRight) / 2.0;
                    foundAll = false;
               }
               else {
                   cells(i).cutMarginLeft = (cells(i).cutMarginLeft + cells(i).cutMarginRight) / 2.0;
                   foundAll = false;
               }
            }
        }

        // Dispatch reshuffle service
        int * dummy;
        std::tie(status, dummy) = MPIMessaging::dispatchService(
                orb,
                localReshuffleService,
                cells(blitz::Range(a, b)).data(),
                b - a,
                dummy,
                b - a,
                std::make_tuple(1,MPIMessaging::np),
                0);

        // Split and store all cells on current heap level
        for (int i = a; i < b; i++) {
            Cell cellLeft;
            Cell cellRight;
            std::tie(cellLeft, cellRight) = cells(i).cut();

            cellRight.setCutAxis();
            cellRight.setCutMargin();
            cellLeft.setCutAxis();
            cellLeft.setCutMargin();

            cells(cells(i).getLeftChildId()) = cellLeft;
            cells(cells(i).getRightChildId()) = cellRight;
        }
    }
}
