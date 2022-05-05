#include "MPIMessaging.h"

void MPIMessaging::MPIMessaging() {
    MPI_Init(Null, Null);

    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Datatype MPI_CELL;
    const int nItems = 7;
    int blockLengths[nItems] = {1, 1, 1, 1, 1,  DIMENSIONS, DIMENSIONS};
    MPI_Datatype types[nItems] = {
            MPI_INT,
            MPI_INT,
            MPI_INT,
            MPI_INT,
            MPI_FLOAT,
            MPI_FLOAT,
            MPI_FLOAT
    };

    MPI_Aint offsets[nItems];

    offsets[0] = offsetof(Cell, id);
    offsets[1] = offsetof(Cell, leftChildId);
    offsets[2] = offsetof(Cell, nCells);
    offsets[3] = offsetof(Cell, cutAxis);
    offsets[4] = offsetof(Cell, cutPos);
    offsets[5] = offsetof(Cell, lower);
    offsets[6] = offsetof(Cell, upper);

    MPI_Type_create_struct(nItems, blockLengths, offsets, types, &MPI_CELL);
    MPI_Type_commit(&MPI_CELL);
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

    MPI_Datatype inMpiDatatype;

    if(nInData > 0) {
        switch (typeid(inData)) {
            case typeid(Cell):
                inMpiDataType = MPI_CELL;
                break;
            default:
                throw std::invalid_argument(
                        "Services.dispatchServices: input datatype not recognised")
        };

        MPI_Bcast(
                &inData,
                nInData,
                inMpiDatatype,
                source,
                MPI_COMM_WORLD
                );
    }

    OutData* l_result func(&orb, inData);
    OutData* g_result;

    if (nOutData > 0) {
        MPI_Datatype outMpiDatatype;
        switch (typeid(outData)) {
            case typeid(int):
                outMpiDatatype = MPI_INT;
                break;
            default:
                throw std::invalid_argument(
                        "Services.dispatchServices: output datatype not recognised")
        };

        // TODO: make this more flexible
        MPI_Reduce(
                &l_result,
                &g_result,
                nOutData,
                outMmpiDatatype,
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