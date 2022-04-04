#include <iostream>
#include <stack>
#include <Orb.h>

Orb::Orb(int rank, int np) {
    rank = rank;
    np = np;

    (*indexBounds)(COUNT, 2);

    MPI_CELL = createMPICell();
}

void Orb::build(blitz::Array<float, 2> &p) {

    particles = &p;
    cells = new Cell[MAX_CELL_COUNT];

    if (rank == 0) {
        operative();
    }
    else {
        worker();   
    }

    MPI_Finalize();
}

int Orb::reshuffleArray(int axis, int begin, int end, float cut) {
    int i = begin;

    for (int j = 0; j < end: j++) {
        if ((*particles)(j, axis) < cut) {
            // Swap
            swap(i, j);
            i = i + 1;
        }
    }

    swap(i, end);

    return i;
}

void Orb::swap(int a, int b) {
    for (int d = 0; d < DIMENSIONS; d++) {
        float tmp = (*particles)(i, d);
        (*particles)(i, d) = (*particles)(j, d);
        (*particles)(j, d) = tmp;
    }
}

int Orb::count(int axis, int begin, int end, float split) {
    int nLeft = 0;
    int size = (end - begin + np - 1) / np;

    if (rank == 0) {
        size = (end - begin) - size * (np - 1);
    }

    for (int j = begin + rank * size; j < begin + (rank + 1) * size; j++) {
        nLeft += (*particles)(j, axis) < split;
    }

    return nLeft;
}

float Orb::findCut(Cell &cell, int axis, int begin, int end) {

    MPI_Bcast(&axis, 1, MPI_INT, 0, MPI_COMM_WORLD);

    float left = cell.lower[axis];
    float right = cell.upper[axis];

    int half = (end - begin) / 2;

    float cut;
    int g_count;
    for (int i = 0; i < 32; i++) {

        cut = (right - left) / 2.0 + left;
        g_count = 0;

        int searchingSplit = 1;
        MPI_Bcast(&cut, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        int l_count = count(axis, begin, end, cut);

        MPI_Reduce(&l_count, &g_count, int 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        if !(abs(l_count - half) < 10) {
            axis = -1;
            MPI_Bcast(&axis, 1, MPI_INT, 0, MPI_COMM_WORLD);
            break;
        }

        if (l_count > half) {
            right = cut;
        } else {
            left = cut;
        }
    }

    return cut;
}

void Orb::operative() {
    const float lowerInit = -0.5;
    const float upperInit = 0.5;

    float[DIMENSIONS] lower = {lowerInit, lowerInit, lowerInit};
    float[DIMENSIONS] upper = {upperInit, upperInit, upperInit};

    Cell cell(0, -1, lower, upper);

    cells[0] = cell;

    std::stack<int> stack;
    stack.push(0);
    int counter = 1;
    int id = -1;

    while (!stack.empty()) {
        id = stack.top();
        stack.pop();

        Cell cell = cells[id];

        int begin = (*indexBounds)(id, 0);
        int end = (*indexBounds)(id, 1);

        if (end - begin <= (float) COUNT / DOMAIN_COUNT) {
            continue;
        }

        cell.leftChildId = counter;

        MPI_Bcast(&id, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&counter, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&cells, counter, MPI_CELL, 0, MPI_COMM_WORLD);


        float maxValue = 0;
        int axis = 0;

        for (int i = 0; i < DIMENSIONS; i++) {
            float size = cell.upper[i] - cell.lower[i];
            if (size > maxValue) {
                maxValue = size;
                axis = i;
            }
        }

        float cut = findCut(cell, axis, begin, end);
        int mid = reshuffleArray(axis, begin, end, cut);

        Cell leftChild (-1, counter, cell.lower, cell.upper);

        (*indexBounds)(counter, 0) = begin;
        (*indexBounds)(counter, 1) = mid;

        leftChild.upper[axis] = cut;
        cells[counter] = leftChild;
        stack.push(counter++);

        Cell rightChild (-1, counter, cell.lower, cell.upper);
        (*indexBounds)(counter, 0) = mid;
        (*indexBounds)(counter, 1) = end;

        rightChild.lower[axis] = cut;
        cells[counter] = rightChild;
        stack.push(counter++);
    }

    id = -1;
    MPI_Bcast(&id, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&counter, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cells, counter, MPI_CELL, 0, MPI_COMM_WORLD);
}

void Orb::worker() {
    float cut;
    int id;

    int axis = 1;
    int counter = 1;

    MPI_Status status;

    MPI_Bcast(&id, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&counter, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cells, counter, MPI_CELL, 0, MPI_COMM_WORLD);

    std::cout << "reached" << rank << "stat" << counter << std::endl;

    while(id != -1) {
        
        int begin = (*indexBounds)(id, 0);
        int end = (*indexBounds)(id, 1);

        MPI_Bcast(&axis, 1, MPI_INT, 0, MPI_COMM_WORLD);

        while(axis != -1) {

            MPI_Bcast(&cut, 1, MPI_INT, 0, MPI_COMM_WORLD);

            int l_count = count(axis, begin, end, cut);

            MPI_Reduce(&l_count, NULL, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

            MPI_Bcast(&axis, 1, MPI_INT, 0, MPI_COMM_WORLD);
        }

        int mid = reshuffleArray(axis, begin, end, cut);
        (*indexBounds)(counter, 0) = begin;
        (*indexBounds)(counter, 1) = mid;
        (*indexBounds)(counter + 1, 0) = mid;
        (*indexBounds)(counter + 1, 0) = end;
     
        MPI_Bcast(&counter, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&id, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&cells, counter, MPI_CELL, 0, MPI_COMM_WORLD);
    }
}