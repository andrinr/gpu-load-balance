#include <stdio.h>
#include <stdlib.h>

static int COUNT = 10 ^ 9;
static int MIN_SIZE = 16;

struct Cell
{
    struct Cell *left;
    struct Cell *right;
    float center [3];
    float size [3];
    int start;
    int end;
};

int main()
{
    // Init positions
    float p [3][COUNT];

    for (int j = 0; j < 3; j++) {
        for (int i = 0; i < COUNT; i++) {
            p[j][i] = (float)rand()/(float)(RAND_MAX) - 0.5;
        }
    }

    // Init tree
    struct Cell root = {
        .center = {0.0, 0.0, 0.0},
        .size = {1.0, 1.0, 1.0},
        .start = 0,
        .end = COUNT
    };
    
    orb(&root, &p);

    return 0;
}

void orb(struct Cell *cell, float *p [3][COUNT]) {

    int axis = findMaxIndex(&cell->size);

    if (cell->end - cell->start < MIN_SIZE){
        return;
    }

    float left = cell->center[axis]-cell->size[axis]/2.0;
    float right = cell->center[axis]+cell->size[axis]/2.0;
    
    float split = findSplit(&p[axis], cell->start, cell->end, left, right);

    struct Cell leftChild = {
        .start
    };

    struct Cell rightChild = {

    };


}

float reshuffleArray(float *arr[], int start, int end, float split) {

}

float findSplit(float *arr[], int start, int end, float left, float right) {
    int half = (end - start) / 2;

    float split = 0;
    for (int i = 0; i < 32; i++) {

        split = (right - left ) / 2.0;

        int nLeft = 0;

        for (int j = start; j < end; j++) {
            nLeft += (*arr)[j] < split;
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
    }

    return split;
}


int findMaxIndex(float *arr[]) {

    float max = 0;
    int index = -1;

    for (int i = 0; i < sizeof(*arr); i++) {
        float element = (*arr)[i];
        if (element > max) {
            index = i;
            max = element;
        };
    };

    return index;
}