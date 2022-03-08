
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <iostream>


static const int DIMENSIONS = 3;

struct Cell
{
    struct Cell *left;
    struct Cell *right;
    std::vector<float> center;
    std::vector<float> size;
    int start;
    int end;
};


int reshuffleArray(std::vector<std::vector<float>> arr, int axis, int start, int end, float split) {
    int i = start;
    int j = end-1;
    
    while (i < j) {
        if (arr[i][axis] < split) {
            i += 1;
        }
        else if (arr[j][axis] > split) {
            j -= 1;
        }
        else {
            for (int d = 0; d < DIMENSIONS; d++) {
                float tmp = arr[i][d];
                arr[i][d] = arr[j][d];
                arr[j][d] = tmp;
            }

            i += 1;
            j -= 1;
        }
    }

    return i;
}

int findMaxIndex(std::vector<float> arr) {

    float max = 0;
    int index = -1;

    for (int i = 0; i < DIMENSIONS; i++) {
        float element = arr[i];
        if (element > max) {
            index = i;
            max = element;
        };
    };

    return index;
}

float findSplit(std::vector<std::vector<float>> arr, int axis, int start, int end, float left, float right) {
    int half = (end - start) / 2;

    float split;
    for (int i = 0; i < 32; i++) {
        split = (right - left ) / 2.0 + left;

        int nLeft = 0;

        for (int j = start; j < end; j++) {
            nLeft += arr[j][axis] < split;
        }
        
        if (abs(nLeft - half) <= 1 ) {
            break;
        }

        if (nLeft > half) {
            right = split;
        } 
        else {
            left = split;
        }
    }
    return split;
}

void orb(struct Cell *cell, std::vector<std::vector<float>> p, int minSize) {

    int axis = findMaxIndex(cell->size);
    if (cell->end - cell->start <= minSize){
       return;
    }

    float left = cell->center[axis]-cell->size[axis]/2.0;
    float right = cell->center[axis]+cell->size[axis]/2.0;
    
    float split = findSplit(p, axis, cell->start, cell->end, left, right);
    int mid = reshuffleArray(p, axis, cell->start, cell->end, split);

    std::vector<float> centerLeft, centerRight, sizeLeft, sizeRight;

    sizeLeft = cell->size;
    sizeRight = cell->size;
    centerLeft = cell->center;
    centerRight = cell->center;

    sizeLeft[axis] = split - left;
    sizeRight[axis] = right - split;

    centerLeft[axis] = left + sizeLeft[axis] / 2.0;
    centerRight[axis] = right - sizeRight[axis] / 2.0;

    struct Cell leftChild = {
        .center = centerLeft,
        .size = sizeLeft,
        .start = cell->start,
        .end = mid,
    };

    struct Cell rightChild = {
        .center = centerLeft,
        .size = sizeLeft,
        .start = mid,
        .end = cell->end
    };

    cell->left = &leftChild;
    cell->right = &rightChild;

    orb(cell->left, p, minSize);
    orb(cell->right, p, minSize);
}
