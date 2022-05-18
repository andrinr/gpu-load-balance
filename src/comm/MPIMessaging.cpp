#include "MPIMessaging.h"

MPIMessaging::MPIMessaging() {
    MPI_Init(NULL, NULL);

    MPI_Comm_size(MPI_COMM_WORLD, &np );
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
}

// Called by worker
void MPIMessaging::workService(ServiceManager * manager) {
    void * rawOutputData;
    workService(manager, rawOutputData);
}

// Called by worker
void MPIMessaging::workService(ServiceManager * manager, void *rawOutputData) {
    int serviceId = -1;
    void * rawInputData;
    dispatchService(manager, serviceId, rawInputData, rawOutputData);
}

// Called by operative
void MPIMessaging::dispatchService(ServiceManager * manager, int serviceID, void *rawInputData, void *rawOutputData) {

    MPI_Bcast(&serviceID, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // Get appropriate service class
    BaseService service = manager->m[id];

    int nInputBytes = service.getNInputBytes(rawInputData);
    MPI_Bcast(&nInputBytes, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int nOutputBytes = service.getNOutputBytes(rawOutputData);
    MPI_Bcast(&nOutputBytes, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Bcast(
            &rawInputData,
            nInputBytes,
            MPI_BYTE,
            0,
            MPI_COMM_WORLD
    );

    service.run(rawInputData, rawOutputData);

    MPI_Bcast(
            &rawOutputData,
            nOutputBytes,
            MPI_BYTE,
            0,
            MPI_COMM_WORLD
    );
}

void MPIMessaging::finalize() {
    MPI_Finalize();
}