#include <iostream>
#include <stack>
#include "mpi.cpp"

static const int DIMENSIONS = 3;
static const int DOMAIN_COUNT = 8;
static const int COUNT = 32;

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

std::tuple<float, int> findSplit(float* arr, int axis, int start, int end, float left, float right) {
    int half = (end - start) / 2;

    float split;
    int nLeft;
    for (int i = 0; i < 32; i++) {
        split = (right - left ) / 2.0 + left;
        nLeft = 0;
        for (int j = start; j < end; j++) {
            nLeft += arr[j * DIMENSIONS + axis] < split;
        }
        std::cout << "nLeft " << nLeft << " " << split << " " << left << " " << right << std::endl;
        
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
    return {split, nLeft};
}

void orb(float* p, int minSize) {

    int pid = mpi::init();

    int counter = 1;
    
    int* leftChild = new int[DOMAIN_COUNT * 2]{0};
    int* begin = new int[DOMAIN_COUNT * 2]{0};
    int* end = new int[DOMAIN_COUNT * 2]{0};
    float* cornerA = new float[DIMENSIONS * DOMAIN_COUNT * 2]{0.0};
    float* cornerB = new float[DIMENSIONS * DOMAIN_COUNT * 2]{0.0};

    cornerA[0] = -0.5;
    cornerA[1] = -0.5;
    cornerA[2] = -0.5;
    cornerB[0] = 0.5;
    cornerB[1] = 0.5;
    cornerB[2] = 0.5;
    begin[0] = 0;
    end[0] = COUNT;

    std::stack<int> stack;
    stack.push(0);
    
    while (!stack.empty()) {
        int id = stack.top();
        stack.pop();

        float maxValue = 0;
        int axis = -1;

        std::cout << "id " << id << std::endl;
        for (int i = 0; i < DIMENSIONS; i++) {
            float size = cornerB[id * DIMENSIONS + i] - cornerA[id * DIMENSIONS + i];
            if (size > maxValue) {
                maxValue = size;
                axis = i;
            }       
        }
        
        std::cout << begin[id] << " " << end[id] << std::endl;

        if (end[id] - begin[id] <= (float) COUNT / DOMAIN_COUNT){
            continue;
        }

        float left = cornerA[id * DIMENSIONS + axis];
        float right = cornerB[id * DIMENSIONS + axis];
        
        float split;
        int mid;
        std::tie(split, mid) =  findSplit(p, axis, begin[id], end[id], left, right);
        mid += begin[id];
        reshuffleArray(p, axis, begin[id], end[id], split);

        leftChild[id] = counter;
        
        // Left Child info
        for (int i = 0; i < DIMENSIONS; i++) {
            cornerA[counter * DIMENSIONS + i] - cornerA[id * DIMENSIONS + i];      
            cornerB[counter * DIMENSIONS + i] - cornerB[id * DIMENSIONS + i];      
        }
        cornerB[counter * DIMENSIONS + axis] = split;
        begin[counter] = begin[id];
        end[counter] = mid;
        stack.push(counter);

        std::cout << counter << " mid " << mid << std::endl;

        counter += 1;

        // Right Child
        for (int i = 0; i < DIMENSIONS; i++) {
            cornerA[counter * DIMENSIONS + i] - cornerA[id * DIMENSIONS + i];      
            cornerB[counter * DIMENSIONS + i] - cornerB[id * DIMENSIONS + i];      
        }
        cornerA[counter * DIMENSIONS + axis] = split;
        begin[counter] = mid;
        end[counter] = end[id];
        stack.push(counter);

        counter += 1;
    }

    mpi::finallize();
}
