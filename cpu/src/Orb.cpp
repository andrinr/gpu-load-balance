#include <iostream>
#include <stack>
#include <Orb.h>

Orb::Orb(blitz::Array<float, 2> &p) {
    particles = particles;
    cells = new Cell[MAX_CELL_COUNT];

    std::cout << "Size:" << (*particles).size() << std::endl;

    MPI_Init(NULL, NULL);

    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* create a type for struct CutDatad */
    const int nitems=4;
    int  blocklengths[4] = {1, 1, 1, 1};
    MPI_Datatype types[4] = {MPI_INT, MPI_INT, MPI_INT, MPI_FLOAT};
    MPI_Datatype mpi_cut_type;
    MPI_Aint offsets[4];

    offsets[0] = offsetof(Cut, begin);
    offsets[1] = offsetof(Cut, end);
    offsets[2] = offsetof(Cut, axis);
    offsets[3] = offsetof(Cut, pos);

    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_cut_type);
    MPI_Type_commit(&mpi_cut_type);
}

void Orb::build() {

    std::cout << "Processor ID: " << rank << " Number of processes: " << np << std::endl;

    if (rank == 0) {
        operative();
    }
    else {
        worker();   
    }

    MPI_Finalize();
}

// You can move them after each split
// Or you have another array to keep track
// Local and global shuffle, ??
// 
void Orb::reshuffleArray(int axis, int begin, int end, float split) {
    int i = begin;
    int j = end-1;

    while (i < j) {
        if ((*particles)(i, axis) < split) {
            i += 1;
        }
        else if ((*particles)(j, axis) > split) {
            j -= 1;
        }
        else {
            for (int d = 0; d < DIMENSIONS; d++) {
                float tmp = (*particles)(i, d);
                (*particles)(i, d) = (*particles)(j, d);
                (*particles)(j, d) = tmp;
            }

            i += 1;
            j -= 1;
        }
    }
}

int Orb::count(int axis, int begin, int end, float split) {
    int nLeft = 0;
    int size = (end - begin + np - 1) / np;

    if (rank == 0) {
        size = (end - begin) - size * (np - 1);
    }

    std::cout << (*particles)(20,0) << std::endl;

    std::cout << "Size:" << (*particles).size() << "Begin:" << begin << "End" << end << std::endl;
    for (int j = begin + rank * size; j < begin + (rank + 1) * size; j++) {
        std::cout << nLeft <<  " " << j << " " << begin + (rank + 1) * size << std::endl;
        nLeft += (*particles)(j, axis) < split;
    }
    //std::cout << "done" << std::endl;
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

        Cut cutdata = {
            begin, axis, end, cut
        };
        
        bool searchingSplit = true;

        std::cout << "Cut:" << cut << std::endl;

        for (int r = 1; r < np; r++) {
            MPI_Send(&cutdata, 1, mpi_cut_type, r,  0, MPI_COMM_WORLD);
        }

        nLeft = count(axis, begin, end, cut);

        for (int r = 1; r < np; r++) {
            int count = 0;
            MPI_Recv(&count, 1, MPI_INT, r,  0, MPI_COMM_WORLD,  MPI_STATUS_IGNORE);
            nLeft += count;
        }

        std::cout << "n:" << nLeft << std::endl;

        searchingSplit = !(abs(nLeft - half) < 10);

        for (int r = 1; r < np; r++) {
            MPI_Send(&searchingSplit, 1, MPI_BOOL, r,  0, MPI_COMM_WORLD);
        }

        if (!searchingSplit) {
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
    Cell cell = Cell{
        0,
        COUNT,
        0,
        -1,
        {-0.5, -0.5, -0.5},
        {0.5, 0.5, 0.5}
    };
    cells[0] = cell;

    std::stack<int> stack;
    stack.push(0);
    int counter = 1;

    while (!stack.empty()) {

        // Broadcast not yet done
        for (int r = 1; r < np; r++) {
            MPI_Send(&true, 1, MPI_BOOL, r,  0, MPI_COMM_WORLD);
        }

        // Broadcast new particle array
        for (int r = 1; r < np; r++) {
            MPI_Send(particles.data(), particles.size(), MPI_FLOAT, r, 0, MPI_COMM_WORLD);
        }

        int id = stack.top();
        stack.pop();

        Cell cell = cells[id];

        std::cout << id << std::endl;
        float maxValue = 0;
        int axis = -1;

        for (int i = 0; i < DIMENSIONS; i++) {
            float size = cell.upper[i] - cell.lower[i];
            if (size > maxValue) {
                maxValue = size;
                axis = i;
            }
        }

        if (cell.end - cell.begin <= (float) COUNT / DOMAIN_COUNT) {
            continue;
        }

        float left = cell.lower[axis];
        float right = cell.upper[axis];

        float cut;
        int mid;
        std::tie(cut, mid) =
                findCut(axis, cell.begin, cell.end, left, right);


        Cell leftChild = {
            cell.begin,
            mid,
            -1,
            counter,
            {0,0,0},
            {0,0,0}
        };
        cells[counter] = leftChild;
        stack.push(counter++);

        Cell rightChild = {
            mid,
            cell.end,
            -1,
            counter,
            {0,0,0},
            {0,0,0}
        };
        cells[counter] = rightChild;
        stack.push(counter++);

        std::cout << id << std::endl;

        std::copy(std::begin(cell.lower), std::end(cell.lower), std::begin(leftChild.lower));
        std::copy(std::begin(cell.lower), std::end(cell.lower), std::begin(rightChild.lower));
        std::copy(std::begin(cell.upper), std::end(cell.upper), std::begin(leftChild.upper));
        std::copy(std::begin(cell.upper), std::end(cell.upper), std::begin(rightChild.upper));

        leftChild.upper[axis] = cut;
        rightChild.lower[axis] = cut;

        reshuffleArray(axis, cell.begin, cell.end, cut);
    }

    for (int r = 1; r < np; r++) {
        MPI_Send(&false, 1, MPI_BOOL, r,  0, MPI_COMM_WORLD);
    }
}

void Orb::worker() {
    float cut;
    int id;
    Cut cutData;

    bool searchingSplit = true;
    bool buildingTree = true;

    while(buildingTree) {
        MPI_RECV(particles.data(), particles.size(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        while(searchingSplit) {
        
            MPI_Recv(&cutData, 1, mpi_cut_type, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            int nLeft = count(cutData.axis, cutData.begin, cutData.end, cutData.pos);

            MPI_Send(&nLeft, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

            MPI_RECV(&searchingSplit, 1, MPI_BOOL, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        }
        MPI_RECV(&buildingTree, 1, MPI_BOOL, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}
