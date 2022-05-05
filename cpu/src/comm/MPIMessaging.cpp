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

int * MPIMessaging::dispatchService(
        Orb &orb,
        int *(*func)(Orb&, Cell *, int),
        Cell *cells,
        int nCells,
        int *results,
        int nResults,
        int source) {

    if(nCells > 0) {
        MPI_Bcast(
                &cells,
                nCells,
                MPI_CELL,
                source,
                MPI_COMM_WORLD
                );
    }

    int* l_result = func(orb, cells, nCells);
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

    return g_result;
}

void MPIMessaging::destroy() {
    MPI_Finalize();
}