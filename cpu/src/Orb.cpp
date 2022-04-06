#include <iostream>
#include <stack>
#include <Orb.h>

MPI_Datatype createCell() {
    MPI_Datatype MPI_CELL;
    const int nitems=4;
    int  blocklengths[nitems] = {1, 1, DIMENSIONS, DIMENSIONS};
    MPI_Datatype types[nitems] = {MPI_INT, MPI_INT, MPI_FLOAT, MPI_FLOAT};
    MPI_Aint offsets[nitems];

    offsets[0] = offsetof(Cell, id);
    offsets[1] = offsetof(Cell, leftChildId);
    offsets[2] = offsetof(Cell, lower);
    offsets[3] = offsetof(Cell, upper);

    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &MPI_CELL);
    MPI_Type_commit(&MPI_CELL);

    return MPI_CELL;
}

Orb::Orb(int r, int n) {
    rank = r;
    np = n;

    MPI_CELL = createCell();
}

void Orb::build(blitz::Array<float, 2> &p) {
    particles = &p;

    std::cout << rank << std::endl;
    if (rank == 0) {
        operative();
    }
    else {
        worker();   
    }
}

int Orb::reshuffleArray(int axis, int begin, int end, float cut) {
    int i = begin;

    for (int j = begin; j < end; j++) {
        if ((*particles)(j, axis) < cut) {
            swap(i, j);
            i = i + 1;
        }
    }

    swap(i, end - 1);

    return i;
}

void Orb::swap(int a, int b) {
    for (int d = 0; d < DIMENSIONS; d++) {
        float tmp = (*particles)(a, d);
        (*particles)(a, d) = (*particles)(b, d);
        (*particles)(b, d) = tmp;
    }
}

int Orb::count(int axis, int begin, int end, float split) {
    int nLeft = 0;

    for (int j = begin; j < end; j++) {
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

        cut = (right + left) / 2.0;
        g_count = 0;

        std::cout << cut << std::endl;

        int searchingSplit = 1;
        MPI_Bcast(&cut, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        int l_count = count(axis, begin, end, cut);

        MPI_Reduce(&l_count, &g_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        if (abs(l_count - half) < 2) {
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

    float lower[DIMENSIONS] = {lowerInit, lowerInit, lowerInit};
    float upper[DIMENSIONS] = {upperInit, upperInit, upperInit};

    Cell cell(0, -1, lower, upper);

    std::cout << cell.lower[0] << ":" << cell.lower[1] << ":" << cell.lower[2] << std::endl;

    cells.push_back(cell);
    
    cellBegin.push_back(0);
    cellEnd.push_back(COUNT);

    std::stack<int> stack;
    stack.push(0);
    int counter = 1;
    int id = -1;

    while (!stack.empty()) {
        id = stack.top();
        stack.pop();

        Cell cell = cells[id];

        int begin = cellBegin[id];
        int end = cellEnd[id];

        if (end - begin <= (float) COUNT / DOMAIN_COUNT) {
            continue;
        }

        std::cout << "id " << id << " begin " << begin << " end " << end <<  std::endl;

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

        std::cout << std::fixed << "Operative: Compute cut between " << cell.lower[axis] << " and " << cell.upper[axis]  << std::endl;
        float cut = findCut(cell, axis, begin, end);
        std::cout << "Found cut at " << cut << std::endl;
        std::cout << "Reshuffle array" << std::endl;
        int mid = reshuffleArray(axis, begin, end, cut);
        std::cout << "Found middle at " << mid << std::endl;

        Cell leftChild (-1, counter, cell.lower, cell.upper);
        
        cellBegin.push_back(begin);
        cellEnd.push_back(mid);

        leftChild.upper[axis] = cut;
        cells.push_back(leftChild);
        stack.push(counter++);

        Cell rightChild (-1, counter, cell.lower, cell.upper);

        cellBegin.push_back(mid);
        cellEnd.push_back(end);

        rightChild.lower[axis] = cut;
        cells.push_back(rightChild);
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

    cellBegin.push_back(0);
    cellEnd.push_back(COUNT);

    MPI_Status status;

    MPI_Bcast(&id, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&counter, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cells, counter, MPI_CELL, 0, MPI_COMM_WORLD);

    while(id != -1) {
        
        int begin = cellBegin[id];
        int end = cellEnd[id];

        MPI_Bcast(&axis, 1, MPI_INT, 0, MPI_COMM_WORLD);

        while(axis != -1) {

            MPI_Bcast(&cut, 1, MPI_INT, 0, MPI_COMM_WORLD);

            int l_count = count(axis, begin, end, cut);

            MPI_Reduce(&l_count, NULL, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

            MPI_Bcast(&axis, 1, MPI_INT, 0, MPI_COMM_WORLD);
        }

        int mid = reshuffleArray(axis, begin, end, cut);
        cellBegin.push_back(begin);
        cellEnd.push_back(mid);
        cellBegin.push_back(mid);
        cellEnd.push_back(end);
     
        MPI_Bcast(&counter, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&id, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&cells, counter, MPI_CELL, 0, MPI_COMM_WORLD);
    }
}