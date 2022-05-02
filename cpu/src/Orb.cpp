#include <iostream>
#include <stack>
#include <memory>
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

Orb::Orb(int r, int n, blitz::Array<float, 2> &p, int d)
    : particles(p), rank(r), np(n), domainCount(d) { //JDP
    MPI_CELL = createCell();
    if (rank == 0) {
        operative();
    }
    else {
        worker();   
    }
}

void Orb::assign(int begin, int end, int id) {
    for (int j = begin; j < end; j++) {
        particles(j, 3) = id;
    }
}

int Orb::reshuffleArray(int axis, int begin, int end, float cut) {
    int i = begin;

    for (int j = begin; j < end; j++) {
        if (particles(j, axis) < cut) {
            swap(i, j);
            i = i + 1;
        }
    }

    swap(i, end - 1);

    return i;
}

void Orb::swap(int a, int b) {
    for (int d = 0; d < DIMENSIONS + 1; d++) {
        float tmp = particles(a, d);
        particles(a, d) = particles(b, d);
        particles(b, d) = tmp;
    }
}

int Orb::count(int axis, int begin, int end, float split, int stride) {
    int nLeft = 0;
    assert(&particles(begin,axis)+1 == &particles(begin+1,axis)); // Column major check
    float *slice = &particles(begin,axis);
    auto p0 = slice;
    auto p3 = slice + (end-begin);
    for(auto p=p0; p<p3; ++p) nLeft += *p < split;
    return nLeft;
}

float Orb::findCut(Cell &cell, int axis, int begin, int end) {

    float left = cell.lower[axis];
    float right = cell.upper[axis];

    int g_rows;
    int l_rows = end - begin;
    
    MPI_Reduce(&l_rows, &g_rows, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    int searchingCut = 1;

    float cut;
    int g_count;
    int i = 0;
    while(true) {
        if (abs(g_count - g_rows / 2) < 16 || i++ == PRECISION) {
            searchingCut = 0;
        }
        //std::cout << i << " " << abs(g_count - g_rows / 2) << " " << cut << std::endl;

        MPI_Bcast(&searchingCut, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (searchingCut == 0) {
            break;
        }

        cut = (right + left) / 2.0;

        MPI_Bcast(&cut, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

        int l_count = count(axis, begin, end, cut, 1);

        g_count = 0;
        MPI_Reduce(&l_count, &g_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        float offset = (g_count - g_rows / 2.0) / g_rows * (cell.upper[axis] - cell.lower[axis]);
        if (g_count > g_rows / 2) {
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

    Cell cell(0, -1, domainCount, lower, upper);

    std::vector<Cell> cells;
    cells.push_back(cell);
    
    cellBegin.push_back(0);
    cellEnd.push_back(particles.rows());

    std::stack<int> stack;
    stack.push(0);

    int buildingTree = 1;

    while (true) {
        if (stack.empty()) {
            buildingTree = 0;
        }
        
        MPI_Bcast(&buildingTree, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (buildingTree == 0) {
            break;
        }

        int id = stack.top();
        stack.pop();

        Cell cell = cells[id];

        int begin = cellBegin[id];
        int end = cellEnd[id];

        cell.leftChildId = cells.size();

        MPI_Bcast(&cell, 1, MPI_CELL, 0, MPI_COMM_WORLD);

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
        int mid = reshuffleArray(axis, begin, end, cut);
        int nCellsLeft = ceil(cell.nCells / 2.0);
        int nCellsRight = cell.nCells - nCellsLeft;
        Cell leftChild (cells.size(), -1, nCellsLeft, cell.lower, cell.upper);
        
        cellBegin.push_back(begin);
        cellEnd.push_back(mid);
        assign(begin, mid, cells.size());

        leftChild.upper[axis] = cut;

        if (nCellsLeft > 1) {
            stack.push(cells.size());
        }
        cells.push_back(leftChild);

        Cell rightChild (cells.size(), -1, nCellsRight, cell.lower, cell.upper);

        cellBegin.push_back(mid);
        cellEnd.push_back(end);
        assign(mid, end, cells.size());

        rightChild.lower[axis] = cut;

        if (nCellsRight > 1) {
            stack.push(cells.size());
        }

        cells.push_back(rightChild);
    }
}

void Orb::worker() {
    int buildingTree = 1;

    cellBegin.push_back(0);
    cellEnd.push_back(particles.rows());

    float lower[DIMENSIONS] = {0., 0., 0.};
    float upper[DIMENSIONS] = {0., 0., 0.};
    Cell cell(
        -1, -1, -1, lower, upper
    );

    while(true) {

        MPI_Bcast(&buildingTree, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (buildingTree == 0) {
            break;
        }

        MPI_Bcast(&cell, 1, MPI_CELL, 0, MPI_COMM_WORLD);

        float cut;
        int begin = cellBegin[cell.id];
        int end = cellEnd[cell.id];
        int axis;
        int searchingCut = 1;

        MPI_Bcast(&axis, 1, MPI_INT, 0, MPI_COMM_WORLD);

        int l_rows = end - begin;
        MPI_Reduce(&l_rows, NULL, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        while(true) {

            MPI_Bcast(&searchingCut, 1, MPI_INT, 0, MPI_COMM_WORLD);

            if (searchingCut == 0) {
                break;
            }

            MPI_Bcast(&cut, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

            int l_count = count(axis, begin, end, cut, 1);

            MPI_Reduce(&l_count, NULL, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        }

        int mid = reshuffleArray(axis, begin, end, cut);
        cellBegin.push_back(begin);
        cellEnd.push_back(mid);
        assign(begin, mid, cell.leftChildId);
        cellBegin.push_back(mid);
        cellEnd.push_back(end);
        assign(mid, end, cell.leftChildId + 1);
    }
}
