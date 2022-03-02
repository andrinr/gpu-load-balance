#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int MIN_SIZE = 128;
static int DIMENSIONS = 3;

struct Cell
{
    struct Cell *left;
    struct Cell *right;
    float center [3];
    float size [3];
    int start;
    int end;
};


int reshuffleArray(float arr[][DIMENSIONS], int axis, int start, int end, float split) {
    int i = start;
    int j = end-1;
    
    while (i < j) {
        //printf("%d, %d, \n", i, j);
        //printf("%.3f, %.3f, %.3f \n", arr[i][axis], arr[j][axis], split);
        if (arr[i][axis] < split) {
            i += 1;
        }
        if (arr[j][axis] > split) {
            j -= 1;
        }
        if (arr[i][axis] > split && arr[j][axis] < split){
            for (int d = 0; d < 3; d++) {
                float tmp = arr[i][d];
                arr[i][d] = arr[j][d];
                arr[j][d] = tmp;
            }

            i += 1;
            j -= 1;
        }
    }

    return i - 1;
}

int findMaxIndex(float arr[DIMENSIONS]) {

    float max = 0;
    int index = -1;

    for (int i = 0; i < DIMENSIONS; i++) {
        float element = (arr)[i];
        if (element > max) {
            index = i;
            max = element;
        };
    };

    return index;
}


float findSplit(float arr[][DIMENSIONS], int axis, int start, int end, float left, float right) {
    int half = (end - start) / 2;

    float split;
    for (int i = 0; i < 32; i++) {
        split = (right - left ) / 2.0 + left;

        int nLeft = 0;

        for (int j = start; j < end; j++) {
            nLeft += arr[j][axis] < split;
        }

        if (abs(nLeft - half) < 1 ) {
            break;
        }

        if (nLeft > half) {
            right = split;
        } 
        else {
            left = split;
        }
        //printf("%.3f, %d, %d \n", split, nLeft, half);
    }
    return split;
}

void orb(struct Cell *cell, float p [][DIMENSIONS]) {

    int axis = findMaxIndex(cell->size);

    if (cell->end - cell->start < MIN_SIZE){
        return;
    }

    float left = cell->center[axis]-cell->size[axis]/2.0;
    float right = cell->center[axis]+cell->size[axis]/2.0;
    
    float split = findSplit(p, axis, cell->start, cell->end, left, right);
    int mid = reshuffleArray(p, axis, cell->start, cell->end, split);

    printf("%d, %d, %d \n", cell->start, cell->end, mid);
    printf("%.3f, %.3f, %.3f \n", left, right, split);

    float centerLeft[DIMENSIONS], centerRight[DIMENSIONS], sizeLeft[DIMENSIONS], sizeRight[DIMENSIONS];

    for (int d = 0; d < DIMENSIONS; d++) {
        sizeLeft[d] = cell->size[d];
        sizeRight[d] = cell->size[d];
        centerRight[d] = cell->center[d];
        centerLeft[d] = cell->center[d];
    }

    sizeLeft[axis] = split - left;
    sizeRight[axis] = right - split;

    centerLeft[axis] = left + sizeLeft[axis] / 2.0;
    centerRight[axis] = right - sizeRight[axis] / 2.0;

    
    struct Cell leftChild = {
        .start = cell->start,
        .end = mid
    };

    struct Cell rightChild = {
        .start = mid,
        .end = cell->end
    };

    memcpy(leftChild.center, centerLeft, 3);
    memcpy(leftChild.size, sizeLeft, 3);
    memcpy(rightChild.center, centerRight, 3);
    memcpy(rightChild.size, sizeRight, 3);

    cell->left = &leftChild;
    cell->right = &rightChild;

    orb(cell->left, p);
    orb(cell->right, p);
}
