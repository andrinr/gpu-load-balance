#include <iostream>
#include <stack>
#include "mpi.cpp"

static const int DIMENSIONS = 3;
static const int MAX_DEPTH = 32;

struct Cell
{
    struct Cell *left;
    struct Cell *right;

    //float* Cell;
    //int left;
    
    float cornerA[DIMENSIONS];
    float cornerB[DIMENSIONS];
    
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

void orb(struct Cell *root, float* p, int minSize) {

    int pid = mpi::init();
    std::stack<Cell*> stack;
    stack.push(root);
    
    while (!stack.empty()) {
        Cell* cell = stack.top();
        stack.pop();

        float maxValue = 0;
        int axis = -1;

        for (int i = 0; i < DIMENSIONS; i++) {
            float size = cell->cornerB[i] - cell->cornerA[i];
            if (size > maxValue) {
                maxValue = size;
                axis = i;
            }       
        }
        
        if (cell->end - cell->start <= minSize){
            continue;
        }

        float left = cell->cornerA[axis];
        float right = cell->cornerB[axis];
        
        float split = findSplit(p, axis, cell->start, cell->end, left, right);
        int mid = reshuffleArray(p, axis, cell->start, cell->end, split);

        std::cout << cell->start << " " << mid << " " << cell->end << std::endl;

        std::cout << cell->cornerA[0] << " " << cell->cornerA[1] << " " << cell->cornerA[2] << std::endl;
        std::cout << cell->cornerB[0] << " " << cell->cornerB[1] << " " << cell->cornerB[2] << std::endl;


        struct Cell leftChild = {
            .start = cell->start,
            .end = mid,
        };

        struct Cell rightChild = {
            .start = mid,
            .end = cell->end
        };
        

        std::copy(std::begin(cell->cornerA), std::end(cell->cornerA), std::begin(leftChild.cornerA));
        std::copy(std::begin(cell->cornerA), std::end(cell->cornerA), std::begin(rightChild.cornerA));
        std::copy(std::begin(cell->cornerB), std::end(cell->cornerB), std::begin(leftChild.cornerB));
        std::copy(std::begin(cell->cornerB), std::end(cell->cornerB), std::begin(rightChild.cornerB));

        std::cout << cell->cornerA[0] << " " << cell->cornerA[1] << " " << cell->cornerA[2] << std::endl;
        std::cout << cell->cornerB[0] << " " << cell->cornerB[1] << " " << cell->cornerB[2] << std::endl;

        leftChild.cornerB[axis] = split;
        rightChild.cornerA[axis] = split;

        cell->left = &leftChild;
        cell->right = &rightChild;
        
        stack.push(&rightChild);
        stack.push(&leftChild);
    }

    mpi::finallize();
}
