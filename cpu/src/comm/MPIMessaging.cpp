#include "MPIMessaging.h"

void MPIMessaging::Init() {
    MPI_Init(NULL, NULL);

    MPI_Comm_size(MPI_COMM_WORLD, &MPIMessaging::np );
    MPI_Comm_rank(MPI_COMM_WORLD, &MPIMessaging::rank);

    MPI_Datatype MPI_CELL;
    const int nItems = 7;
    int blockLengths[nItems] = {
            1,
            1,
            1,
            1,
            1,
            DIMENSIONS,
            DIMENSIONS};

    MPI_Datatype types[nItems] = {
            MPI_INT,
            MPI_INT,
            MPI::UNSIGNED_SHORT,
            MPI_FLOAT,
            MPI_FLOAT,
            MPI_FLOAT,
            MPI_FLOAT
    };

    MPI_Aint offsets[nItems];

    offsets[0] = offsetof(Cell, id);
    offsets[1] = offsetof(Cell, nLeafCells);
    offsets[2] = offsetof(Cell, cutAxis);
    offsets[3] = offsetof(Cell, cutMarginLeft);
    offsets[4] = offsetof(Cell, cutMarginRight);
    offsets[5] = offsetof(Cell, lower);
    offsets[6] = offsetof(Cell, upper);

    MPI_Type_create_struct(nItems, blockLengths, offsets, types, &MPIMessaging::MPI_CELL);
    MPI_Type_commit(&MPIMessaging::MPI_CELL);
}

void MPIMessaging::signalServiceId(int id) {
    MPI_Bcast(
            &id,
            1,
            MPI_INT,
            0,
            MPI_COMM_WORLD
    );
}

void MPIMessaging::signalDataSize(int size) {
    MPI_Bcast(
            &size,
            1,
            MPI_INT,
            0,
            MPI_COMM_WORLD
    );
}

std::tuple<bool, int*> MPIMessaging::dispatchService(
        Orb &orb,
        ServiceIDs id,
        Cell *cells,
        int nCells,
        int *results,
        int nResults,
        int source) {

    MPI_Bcast(
            &nCells,
            1,
            MPI_INT,
            source,
            MPI_COMM_WORLD
    );

    if(nCells > 0) {
        MPI_Bcast(
                &cells,
                nCells,
                MPI_CELL,
                source,
                MPI_COMM_WORLD
                );
    }

    MPI_Bcast(
            &id,
            1,
            MPI_INT,
            source,
            MPI_COMM_WORLD
    );

    int* l_result;
    switch (id) {
        case countLeftService:
            l_result = Services::countLeft(orb, cells, nCells);
            break;
        case countService:
            l_result = Services::count(orb, cells, nCells);
            break;
        case buildTreeService:
            l_result = Services::buildTree(orb, cells, nCells);
            break;
        case localReshuffleService:
            l_result = Services::localReshuffle(orb, cells, nCells);
            break;
        case terminateService:
            return std::make_tuple(false, nullptr);
        default:
            throw std::invalid_argument("MPIMessaging.dispatchService: is is unknown.");
    }

    MPI_Bcast(
            &nResults,
            1,
            MPI_INT,
            source,
            MPI_COMM_WORLD
    );

    int* g_result;

    if (nResults > 0) {
        // TODO: make this more flexible
        MPI_Reduce(
                &l_result,
                &g_result,
                nResults,
                MPI_INT,
                MPI_SUM,
                source,
                MPI_COMM_WORLD
                );
    }

    return std::make_tuple(true, g_result);
}

void MPIMessaging::destroy() {
    MPI_Finalize();
}