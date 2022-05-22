//
// Created by andrin on 19/05/22.
//

#include "iostream"
#include "buildTreeService.h"
#include "countLeftService.h"
#include "countService.h"
#include "localReshuffleService.h"


void BuildTreeService::run(void *inputBuffer, int inputBufferLength, void *outputBuffer, int outputBufferLength) {

    std::cout << "building tree" << std::endl;
    BuildTreeServiceInput inputData = *(struct BuildTreeServiceInput*)rawInputData;
    void * outputData;

    Orb orb = *manager->orb;

    // Blitz: 2.3.7
    blitz::Array<Cell, 1> cells(inputData.root, blitz::shape(inputData.root->nLeafCells));

    // loop over levels of tree
    CellHelpers::log(*inputData.root);

    for (int l = 1; l < CellHelpers::getNLevels(*inputData.root); ++l) {

        int a = std::pow(2, (l-1)) - 1;
        // todo: correct this
        int b = std::min(
                CellHelpers::getNCellsOnLastLevel(*inputData.root),
                inputData.root->nLeafCells) - 1;

        std::cout << l << "-level" << " a "  << a << " b " << b << std::endl;

        bool status;
        int* sum;

        CountServiceInput csi {
                cells(blitz::Range(a, b)).data(),
                b - a
        };
        CountServiceOutput * cso;
        inputData.messaging->dispatchService(
                manager,
                COUNT_SERVICE_ID,
                &csi,
                cso
                );
        blitz::Array<int, 1> countBlitz(cso->sums, blitz::shape(b - a));

        // Loop
        bool foundAll = true;

        while(foundAll) {
            int* sumLeft;
            foundAll = true;

            CountLeftServiceInput clsi {
                    cells(blitz::Range(a, b)).data(),
                    b - a
            };
            CountLeftServiceOutput * clso;
            inputData.messaging->dispatchService(
                    manager,
                    COUNT_LEFT_SERVICE_ID,
                    &clsi,
                    clso
            );
            blitz::Array<int, 1> countLeftBlitz(clso->counts, blitz::shape(b - a));

            for (int i = a; i < b; ++i) {

                if (abs(countLeftBlitz(i) - countBlitz(i) / 2.0) < CUTOFF) {
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

        LocalReshuffleServiceInput lrsi {
                cells(blitz::Range(a, b)).data(),
                b - a
        };
        LocalServiceServiceOutput * lrso;
        inputData.messaging->dispatchService(
                manager,
                LOCAL_RESHUFFLE_SERVICE_ID,
                &lrsi,
                lrso
        );

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


std::tuple<int, int> BuildTreeService::getNBytes(int bufferLength) const {
    return std::make_tuple(sizeof Cell, 0);
}

int BuildTreeService::getNInputBytes(void *inputPtr) const {
    // simply two pointers
    return sizeof(Cell);
}

int BuildTreeService::getNOutputBytes(void *outputPtr) const {
    return 0;
}