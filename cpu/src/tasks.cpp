#include <tasks.h>
#include <tasks.h>
#include "communication/mpi-comm.h"

void Tasks::operate(Orb& orb)
    Cell cells;
    cells.push_back(cell);

    std::stack<int> stack;
    stack.push(0);

    int level = 1;

    while (true) {
        int startIndex = level++;

        int id = stack.top();
        stack.pop();

        Cell cell = cells[id];

        int begin = cellBegin[id];
        int end = cellEnd[id];

        cell.leftChildId = cells.size();

        MPI_Bcast(&cell, 1, MPI_CELL, 0, MPI_COMM_WORLD);

        float maxValue = 0;
        int axis = 0;

        for (int i = 0; i < DIMENSIONS; i++) {
            float size = cell.upper[i] - cell.lower[i];
            if (size > maxValue) {
                maxValue = size;
                axis = i;
            }
        }

        MPI_Bcast(&axis, 1, MPI_INT, 0, MPI_COMM_WORLD);

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

void Tasks::work(Orb& orb) {
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

