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

    float left = cell.lower[axis];
    float right = cell.upper[axis];

    int g_rows;
    int l_rows = end - begin;
    
    MPI_Reduce(&l_rows, &g_rows, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    std::cout << g_rows << std::endl;
    int half = (end - begin) / 2;

    int searchingCut = 1;

    float cut;
    int g_count;
    for (int i = 0; i < 32; i++) {

        cut = (right + left) / 2.0;
        g_count = 0;

        MPI_Bcast(&cut, 1, MPI_INT, 0, MPI_COMM_WORLD);

        int l_count = count(axis, begin, end, cut);

        MPI_Reduce(&l_count, &g_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        std::cout << "op " << cut << " count " << g_count << std::endl;

        if (abs(g_count - half) < 1000) {
            searchingCut = 0;
            MPI_Bcast(&searchingCut, 1, MPI_INT, 0, MPI_COMM_WORLD);
            break;
        }

        MPI_Bcast(&searchingCut, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (l_count > half) {
            right = cut;
        } else {
            left = cut;
        }
    }

    return cut;
}

void Orb::operative() {
    // init
    const float lowerInit = -0.5;
    const float upperInit = 0.5;

    float lower[DIMENSIONS] = {lowerInit, lowerInit, lowerInit};
    float upper[DIMENSIONS] = {upperInit, upperInit, upperInit};

    Cell cell(0, -1, lower, upper);

    cells.push_back(cell);
    
    cellBegin.push_back(0);
    cellEnd.push_back((*particles).rows());

    std::stack<int> stack;
    stack.push(0);

    int buildingTree = 1;

    while (!stack.empty()) {
        int id = stack.top();
        stack.pop();

        Cell cell = cells[id];

        int begin = cellBegin[id];
        int end = cellEnd[id];

        if (end - begin <= (float) (*particles).rows() / DOMAIN_COUNT) {
            continue;
        }

        cell.leftChildId = cells.size();

        MPI_Bcast(&buildingTree, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&cells[id], 1, MPI_CELL, 0, MPI_COMM_WORLD);

        float maxValue = 0;
        int axis = 0;

        for (int i = 0; i < DIMENSIONS; i++) {
            float size = cell.upper[i] - cell.lower[i];
            if (size > maxValue) {
                maxValue = size;
                axis = i;
            }
        }

        MPI_Bcast(&axis, 1, MPI_INT, 0, MPI_COMM_WORLD);
        float cut = findCut(cell, axis, begin, end);
        std::cout << "Operative: Found cut at " << cut << " on axis " << axis << std::endl;
        int mid = reshuffleArray(axis, begin, end, cut);

        Cell leftChild (cells.size(), -1, cell.lower, cell.upper);
        
        cellBegin.push_back(begin);
        cellEnd.push_back(mid);

        leftChild.upper[axis] = cut;
        stack.push(cells.size());
        cells.push_back(leftChild);

        Cell rightChild (cells.size(), -1, cell.lower, cell.upper);

        cellBegin.push_back(mid);
        cellEnd.push_back(end);

        rightChild.lower[axis] = cut;
        stack.push(cells.size());
        cells.push_back(rightChild);
    }

    buildingTree = 0;
    MPI_Bcast(&buildingTree, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void Orb::worker() {
    int buildingTree = 1;

    cellBegin.push_back(0);
    cellEnd.push_back((*particles).rows());

    float lower[DIMENSIONS] = {0., 0., 0.};
    float upper[DIMENSIONS] = {0., 0., 0.};
    Cell cell(
        -1, -1, lower, upper
    );

    MPI_Bcast(&buildingTree, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cell, 1, MPI_CELL, 0, MPI_COMM_WORLD);

    while(buildingTree == 1) {
        float cut;
        int begin = cellBegin[cell.id];
        int end = cellEnd[cell.id];
        int axis;
        int searchingCut = 1;

        MPI_Bcast(&axis, 1, MPI_INT, 0, MPI_COMM_WORLD);

        std::cout << "worker " << axis << std::endl;

        int l_rows = end - begin;
        MPI_Reduce(&l_rows, NULL, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        while(searchingCut == 1) {

            MPI_Bcast(&cut, 1, MPI_INT, 0, MPI_COMM_WORLD);

            int l_count = count(axis, begin, end, cut);
            MPI_Reduce(&l_count, NULL, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

            MPI_Bcast(&searchingCut, 1, MPI_INT, 0, MPI_COMM_WORLD);
        }

        int mid = reshuffleArray(axis, begin, end, cut);
        cellBegin.push_back(begin);
        cellEnd.push_back(mid);
        cellBegin.push_back(mid);
        cellEnd.push_back(end);
     
        MPI_Bcast(&buildingTree, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (buildingTree == 0){
            break;
        }
        MPI_Bcast(&cells, 1, MPI_CELL, 0, MPI_COMM_WORLD);
    }
}