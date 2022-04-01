#include <iostream>
#include <stack>
#include <Orb.h>

Orb::Orb(int rank, int np) {
    rank = rank;
    np = np;

    MPI_CELL = createMPICell();
    MPI_CUT = createMPICut();
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

    for (int j = begin + rank * size; j < begin + (rank + 1) * size; j++) {
        nLeft += (*particles)(j, axis) < split;
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
    int g_count;
    for (int i = 0; i < 32; i++) {

        cut = (right - left) / 2.0 + left;
        g_count = 0;

        Cut cutdata = {
            begin, axis, end, cut
        };
        
        int searchingSplit = 1;

        MPI_Bcast(&cutdata, 1, mpi_cut_type, 0, MPI_COMM_WORLD);
        
        int l_count = count(axis, begin, end, cut);

        MPI_Reduce(
            &l_count,
            &g_count,
            int 1,
            MPI_INT,
            MPI_SUM,
            0,
            MPI_COMM_WORLD);

        std::cout << "n:" << l_count << std::endl;

        searchingSplit = int(!(abs(l_count - half) < 10));

        std::cout << "searching" << searchingSplit << std::endl;

        for (int r = 1; r < np; r++) {
            MPI_Send(&searchingSplit, 1, MPI_INT, r,  TAG_CUT_STATUS, MPI_COMM_WORLD);
        }

        if (searchingSplit == 0) {
            break;
        }

        if (l_count > half) {
            right = cut;
        } else {
            left = cut;
        }
    }

    return {cut, g_count + begin};
}

void Orb::operative() {
    const float lowerInit = -0.5;
    const float upperInit = 0.5;

    Cell cell = Cell{
        0,
        COUNT,
        0,
        -1,
    };


    blitz::TinyVector<float, DIMENSIONS> lower;
    blitz::TinyVector<float, DIMENSIONS> upper;

    cell.lower = lowerInit;
    cell.upper = upperInit;

    cells[0] = cell;

    std::stack<int> stack;
    stack.push(0);
    int counter = 1;

    bool buildingTree = 1;

    while (!stack.empty()) {

        std::cout << "next iteration" << rank << std::endl;


        MPI_Bcast(&buildingTree, 1, MPI_INT, 0, MPI_COMM_WORLD);

        int id = stack.top();
        stack.pop();

        Cell cell = cells[id];

        float maxValue = 0;
        int axis = 0;

        for (int i = 0; i < DIMENSIONS; i++) {
            float size = cell.upper(i) - cell.lower(i);
            if (size > maxValue) {
                maxValue = size;
                axis = i;
            }
        }

        if (cell.end - cell.begin <= (float) COUNT / DOMAIN_COUNT) {
            // flawed logic?
            continue;
        }

        float left = cell.lower(axis);
        float right = cell.upper(axis);

        float cut;
        int mid;
        std::tie(cut, mid) = findCut(axis, cell.begin, cell.end, left, right);

        Cell leftChild = {
            cell.begin,
            mid,
            -1,
            counter,
            cell.lower,
            cell.upper
        };
        leftChild.upper(axis) = cut;
        cells[counter] = leftChild;
        cell.leftChildId = counter;
        stack.push(counter++);

        Cell rightChild = {
            mid,
            cell.end,
            -1,
            counter,
            cell.lower,
            cell.upper
        };
        rightChild.lower(axis) = cut;
        cells[counter] = rightChild;
        stack.push(counter++);

        std::cout << id << " axis:" << axis << std::endl;

        reshuffleArray(axis, cell.begin, cell.end, cut);
    }

    buildingTree = 0;
    MPI_Bcast(&buildingTree, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void Orb::worker() {
    float cut;
    int id;
    Cut cutData;

    int searchingSplit = 1;
    int buildingTree = 1;

    MPI_Status status;

    MPI_Bcast(&buildingTree, 1, MPI_INT, 0, MPI_COMM_WORLD);


    std::cout << "reached" << rank << "stat" << buildingTree << std::endl;

    while(buildingTree == 1) {

        while(searchingSplit == 1) {
        
            MPI_Recv(
                &cutData, 
                1, 
                mpi_cut_type, 
                0, 
                TAG_CUT_DATA, 
                MPI_COMM_WORLD, 
                MPI_STATUS_IGNORE
            );

            int nLeft = count(cutData.axis, cutData.begin, cutData.end, cutData.pos);

            MPI_Send(
                &nLeft, 
                1, 
                MPI_INT, 
                0, 
                TAG_CUT_COUNT,
                MPI_COMM_WORLD
            );


            MPI_Reduce(
                &l_count,
                NULL,
                int 1,
                MPI_INT,
                MPI_SUM,
                0,
                MPI_COMM_WORLD);

            MPI_Recv(
                &searchingSplit, 
                1, 
                MPI_INT, 
                0, 
                TAG_CUT_STATUS, 
                MPI_COMM_WORLD, 
                MPI_STATUS_IGNORE
            );

        }

        std::cout << "found split" << rank << std::endl;

        MPI_Recv(
            &buildingTree, 
            1, 
            MPI_INT, 
            0, 
            TAG_TREE_STATUS, 
            MPI_COMM_WORLD, 
            &status
        );

        std::cout <<
         "rank " << rank << 
         "state" << buildingTree << 
         "mpi stat" << status.MPI_SOURCE << 
         "mpi tag" << status.MPI_TAG << std::endl;
    }
}   