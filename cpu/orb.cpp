#include <vector>
#include <iostream>
#include "mpi.cpp"

static const int DIMENSIONS = 3;

struct Cell
{
    struct Cell *left;
    struct Cell *right;
    float center[DIMENSIONS];
    float size[DIMENSIONS];
    
    int start;
    int end;
};


int reshuffleArray(float* arr, int axis, int start, int end, float split) {
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

    return i;
}

int findMaxIndex(float* arr) {

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

float findSplit(float* arr, int axis, int start, int end, float left, float right) {
    int half = (end - start) / 2;

    float split;
    for (int i = 0; i < 32; i++) {
        split = (right - left ) / 2.0 + left;

        int nLeft = 0;

        for (int j = start; j < end; j++) {
            nLeft += arr[j * DIMENSIONS + axis] < split;
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

void orb(struct Cell *cell, float* p, int minSize) {

    int axis = findMaxIndex(cell->size);
    if (cell->end - cell->start <= minSize){
        std::cout << cell->start << " " << cell->end << std::endl;
       return;
    }

    float left = cell->center[axis]-cell->size[axis]/2.0;
    float right = cell->center[axis]+cell->size[axis]/2.0;
    
    float split = findSplit(p, axis, cell->start, cell->end, left, right);
    int mid = reshuffleArray(p, axis, cell->start, cell->end, split);

    float centerLeft[DIMENSIONS]{0.0};
    float centerRight[DIMENSIONS]{0.0};
    float sizeLeft[DIMENSIONS]{0.0};
    float sizeRight[DIMENSIONS]{0.0};


    struct Cell leftChild = {
        .start = cell->start,
        .end = mid,
    };

    struct Cell rightChild = {
        .start = mid,
        .end = cell->end
    };

    std::copy(std::begin(cell->size), std::end(cell->size), std::begin(leftChild.size));
    std::copy(std::begin(cell->size), std::end(cell->size), std::begin(rightChild.size));
    std::copy(std::begin(cell->center), std::end(cell->center), std::begin(leftChild.center));
    std::copy(std::begin(cell->center), std::end(cell->center), std::begin(rightChild.center));


    leftChild.size[axis] = split - left;
    rightChild.size[axis] = right - split;

    leftChild.center[axis] = left + leftChild.size[axis] / 2.0;
    rightChild.center[axis] = right - rightChild.size[axis] / 2.0;

    cell->left = &leftChild;
    cell->right = &rightChild;

    orb(cell->left, p, minSize);
    orb(cell->right, p, minSize);
}
