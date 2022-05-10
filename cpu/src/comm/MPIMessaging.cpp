#include "MPIMessaging.h"

void MPIMessaging::MPI_Handler_function() {

};

void MPIMessaging::Init() {

    MPI_Init(NULL, NULL);

    MPI_Comm_size(MPI_COMM_WORLD, &MPIMessaging::np );
    MPI_Comm_rank(MPI_COMM_WORLD, &MPIMessaging::rank);

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
            MPI_INT,
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

    std::cout << "MpiMessaging initialized" << std::endl;
}

std::tuple<bool, int*> MPIMessaging::dispatchService(
        Orb &orb,
        ServiceIDs id,
        Cell *cells,
        int nCells,
        int *results,
        int nResults,
        int target,
        int source) {

#if DEBUG
    if (source < 0 || source >= MPIMessaging::np) {
        throw std::invalid_argument("MPIMessaging.dispatchService: source is out of bounds.");
    }

    if (target < 0 || target >= MPIMessaging::np) {
        throw std::invalid_argument("MPIMessaging.dispatchService: source is out of bounds.");
    }
#endif // DEBUG

    std::cout << "dispatching " << id << std::endl;

    if (target != source) {
        MPI_Bcast(
                &nCells,
                1,
                MPI_INT,
                source,
                MPI_COMM_WORLD
        );

        if (nCells > 0) {
            MPI_Bcast(
                    &cells,
                    nCells,
                    MPIMessaging::MPI_CELL,
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
    }


    int* l_result;
    std::cout << id << std::endl;
    switch (id) {
        case countLeftService:
            Services::countLeft(orb, cells, l_result, nCells);
            break;
        case countService:
            Services::count(orb, cells, l_result, nCells);
            break;
        case buildTreeService:
            Services::buildTree(orb, cells, l_result, nCells);
            break;
        case localReshuffleService:
            Services::localReshuffle(orb, cells, l_result, nCells);
            break;
        case terminateService:
            return std::make_tuple(false, nullptr);
        default:
            throw std::invalid_argument("MPIMessaging.dispatchService: is is unknown.");
    }

    int *g_result;
    if (target != source) {
        MPI_Bcast(
                &nResults,
                1,
                MPI_INT,
                source,
                MPI_COMM_WORLD
        );

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
    }

    return std::make_tuple(true, g_result);
}

std::tuple<bool, int*> MPIMessaging::dispatchService(
        Orb &orb,
        ServiceIDs id,
        Cell *cells,
        int nCells,
        int *results,
        int nResults,
        std::tuple<int, int> target,
        int source) {

#if DEBUG
    if (source < 0 || source >= MPIMessaging::np) {
        throw std::invalid_argument("MPIMessaging.dispatchService: source is out of bounds.");
    }

    if (source < 0 || source >= MPIMessaging::np) {
        throw std::invalid_argument("MPIMessaging.dispatchService: source is out of bounds.");
    }
#endif // DEBUG

    std::cout << "dispatching " << id << std::endl;

    MPI_Bcast(
            &nCells,
            1,
            MPI_INT,
            source,
            MPI_COMM_WORLD
    );

    std::cout << "n " << nCells << std::endl;

    if(nCells > 0) {
        std::cout << "init cells bcast" << MPIMessaging::MPI_CELL << std::endl;
        MPI_Bcast(
                &cells,
                nCells,
                MPIMessaging::MPI_CELL,
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
    std::cout << id << std::endl;
    switch (id) {
        case countLeftService:
            Services::countLeft(orb, cells, l_result, nCells);
            break;
        case countService:
            Services::count(orb, cells, l_result, nCells);
            break;
        case buildTreeService:
            Services::buildTree(orb, cells, l_result, nCells);
            break;
        case localReshuffleService:
            Services::localReshuffle(orb, cells, l_result, nCells);
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

void MPIMessaging::finalize() {
    MPI_Finalize();
}