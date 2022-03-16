#include <iostream>
#include <stack>
#include <mpi.h>
#include "Orb.h"
#include "Cells.h"

Orb::Orb(float* particles) {
    particles = particles;
}

void Orb::build() {

    int np, rank;
    MPI_Init(NULL, NULL);

    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::cout << "Processor ID: " << rank << " Number of processes: " << np << std::endl;

    if (rank == 0) {
        operative();
    }
    else {
        worker();   
    }

    MPI_Finalize();
}

void Orb::reshuffleArray(int axis, int begin, int end, float split) {
    int i = begin;
    int j = end-1;

    while (i < j) {
        if (particles[i * DIMENSIONS + axis] < split) {
            i += 1;
        }
        else if (particles[j * DIMENSIONS + axis] > split) {
            j -= 1;
        }
        else {
            for (int d = 0; d < DIMENSIONS; d++) {
                float tmp = particles[i * DIMENSIONS + d];
                particles[i * DIMENSIONS + d] = particles[j * DIMENSIONS + d];
                particles[j * DIMENSIONS + d] = tmp;
            }

            i += 1;
            j -= 1;
        }
    }
}

int Orb::countLeft(int axis, int begin, int end, float split) {
    int nLeft = 0;
    for (int j = begin; j < end; j++) {
        nLeft += particles[j * DIMENSIONS + axis] < split;
    }
    return nLeft;
}

std::tuple<float, int> Orb::findCut(
    int axis,
    int begin,
    int end,
    float left,
    float right) {

    int half = (end - begin) / 2;

    float cut;
    int nLeft;
    for (int i = 0; i < 32; i++) {

        cut = (right - left) / 2.0 + left;
        nLeft = 0;
        
       // MPI_SEND(&cut, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

        //countLeft(arr,)

        if (abs(nLeft - half) < 1) {
            break;
        }

        if (nLeft > half) {
            right = cut;
        } else {
            left = cut;
        }
    }

    return {cut, nLeft + begin};
}

void Orb::operative() {
    Cells cells = Cells(-0.5, 0.5);
    std::stack<int> stack;
    stack.push(0);
    int counter = 1;
    while (!stack.empty()) {
        int id = stack.top();
        stack.pop();

        std::cout << id << std::endl;
        float maxValue = 0;
        int axis = -1;


        for (int i = 0; i < DIMENSIONS; i++) {
            float size = cells.getCornerB(id, i) - cells.getCornerA(id, i);
            if (size > maxValue) {
                maxValue = size;
                axis = i;
            }
        }

        if (cells.getEnd(id) - cells.getBegin(id) <= (float) COUNT / DOMAIN_COUNT) {
            continue;
        }

        float left = cells.getCornerA(id, axis);
        float right = cells.getCornerB(id, axis);

        float cut;
        int mid;
        std::tie(cut, mid) =
                findCut(axis, cells.getBegin(id), cells.getEnd(id), left, right);

        // Copy data
        for (int i = 0; i < DIMENSIONS; i++) {
            cells.setCornerA(counter, i,  cells.getCornerA(id, i));
            cells.setCornerB(counter, i,  cells.getCornerB(id, i));
            cells.setCornerA(counter + 1, i,  cells.getCornerA(id, i));
            cells.setCornerB(counter + 1, i,  cells.getCornerB(id, i));
        }

        cells.setCornerB(counter, axis, cut);
        cells.setBegin(counter, cells.getBegin(id));
        cells.setEnd(counter, mid);

        stack.push(counter++);

        cells.setCornerA(counter + 1, axis, cut);
        cells.setBegin(counter, mid);
        cells.setEnd(counter, cells.getEnd(id));

        stack.push(counter++);

        reshuffleArray(axis, cells.getBegin(id), cells.getEnd(id), cut);
    }
}

void Orb::worker() {

}
