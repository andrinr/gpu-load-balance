//
// Created by andrin on 19/05/22.
//

#include "buildTreeService.h"

void BuildTreeService::run(void *rawInputData, void *rawOutputData) {

    ServiceInput inputData = *(struct ServiceInput*)rawInputData;
    ServiceOutput outputData;

    Orb orb = *manager->orb;

    // Blitz: 2.3.7
    blitz::Array<Cell, 1> cells(c, blitz::shape(n));

    // loop over levels of tree
    CellHelpers::log(root);

    for (int l = 1; l < ceil(log2(inputData.root->nLeafCells)); ++l) {

        int a = std::pow(2, (l-1)) - 1;
        // todo: correct this
        int b = std::min((int)std::pow(2, l), inputData.root->nLeafCells) - 1;

        std::cout << l << "-level" << " a "  << a << " b " << b << std::endl;

        bool status;
        int* sum;

        inputData.messaging->dispatchService(
                
                )
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
            std::tie(status, sumLeft) = mpiMessaging.dispatchService(
                    orb,
                    countLeftService,
                    cells(blitz::Range(a, b)).data(),
                    b - a,
                    sumLeft,
                    b - a,
                    std::make_tuple(1,MPIMessaging::np),
                    0);

            blitz::Array<int, 1> countLeftBlitz(sumLeft, blitz::shape(b - a));

            for (int i = a; i < b; ++i) {

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
        std::tie(status, dummy) = mpiMessaging.dispatchService(
                orb,
                localReshuffleService,
                cells(blitz::Range(a, b)).data(),
                b - a,
                dummy,
                b - a,
                std::make_tuple(1,MPIMessaging::np),
                0);

        // Split and store all cells on current heap level
        for (int i = a; i < b; ++i) {
            Cell cellLeft;
            Cell cellRight;
            std::tie(cellLeft, cellRight) = CellHelpers::cut(cells(i));

            CellHelpers::setCutAxis(cellRight);
            CellHelpers::setCutAxis(cellLeft);
            CellHelpers::setCutMargin(cellLeft);
            CellHelpers::setCutMargin(cellRight);

            cells(CellHelpers::getLeftChildId(cells(i))) = cellLeft;
            cells(CellHelpers::getRightChildId(cells(i))) = cellRight;
        }
    }
}