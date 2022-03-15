#include <iostream>
#include <stack>
#include "cells.h"
#include "mpi.cpp"  

void reshuffleArray(float* arr, int axis, int start, int end, float split) {
    int i = start;
    int j = end-1;
    
    while (i < j) {
        if (arr[i * DIMENSIONS + axis] < split) {
            i += 1;
        }
        else if (arr[j * DIMENSIONS + axis] > split) {
            j -= 1;
        }
        else {
            for (int d = 0; d < DIMENSIONS; d++) {
                float tmp = arr[i * DIMENSIONS + d];
                arr[i * DIMENSIONS + d] = arr[j * DIMENSIONS + d];
                arr[j * DIMENSIONS + d] = tmp;
            }

            i += 1;
            j -= 1;
        }
    }
}

int countLeft(float* arr, int start, int end, float split) {
    int nLeft = 0;
    for (int j = start; j < end; j++) {
        nLeft += arr[j * DIMENSIONS + axis] < split;
    }
}

std::tuple<float, int> findCut(
        float* arr,
        int axis,
        int start,
        int end,
        float left,
        float right,
        int rank ) {

    int half = (end - start) / 2;

    float split;
    int nLeft;
    for (int i = 0; i < 32; i++) {

        if (rank == 0) {
            split = (right - left ) / 2.0 + left;
            nLeft = 0;
        }

        MPI_SEND(&split, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

        countLeft(arr, )

        if (rank == 0) {
            if (abs(nLeft - half) < 1 ) {
                break;
            }

            if (nLeft > half) {
                right = split;
            } 
            else {
                left = split;
            }
        }
    }

    return {split, nLeft};
}

void operative() {


    Cells cells = new Cells();
    cells.init();

    int childId = 1;
    while (!stack.empty()) {
        int id = stack.top();
        stack.pop();

        std::cout << id << std::endl;
        float maxValue = 0;
        int axis = -1;


        for (int i = 0; i < DIMENSIONS; i++) {
            float size = cornerB[id * DIMENSIONS + i] - cornerA[id * DIMENSIONS + i];
            if (size > maxValue) {
                maxValue = size;
                axis = i;
            }
        }

        if (end[id] - begin[id] <= (float) COUNT / DOMAIN_COUNT) {
            continue;
        }

        float left = cornerA[id * DIMENSIONS + axis];
        float right = cornerB[id * DIMENSIONS + axis];
        
        float split;
        int mid;
        std::tie(split, mid) = findCut(p, axis, begin[id], end[id], left, right);
        mid += begin[id];
        reshuffleArray(p, axis, begin[id], end[id], split);
    }
}

void worker() {

}

void orb(float* p, int minSize) {

    int rank, np;
    std::tie(split, mid) = mpi::init();

    if (rank == 0) {
        operative();
    } 
    else {
        worker();
    }

    mpi::finallize();
}
