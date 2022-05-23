#include "MPIMessaging.h"
#include "../services/baseService.h"

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
void MPIMessaging::workService(ServiceManager * manager, void * outputBuffer) {
    int serviceId = -1;
    void * inputBuffer;
    int nInputBufferBytes;
    void * outputBuffer;
    int nOutputBufferBytes;
    dispatchService(manager, serviceId, inputBuffer, nInputBufferBytes, outputBuffer, nOutputBufferBytes);
}

// Called by operative
// We assume the operative knows about the input and output data exactly
void MPIMessaging::dispatchService(
        ServiceManager * manager,
        int serviceID,
        void * inputBuffer,
        int nInputBufferBytes,
        void * outputBuffer,
        int nOutputBufferBytes,
        int source,
        std::tuple<int, int> target) {

    MPI_Bcast(&serviceID, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // Get appropriate service class
    std::cout << "dispatching service " << serviceID << std::endl;

    int nInputBytes, nOutputBytes;

    MPI_Bcast(&nInputBytes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nOutputBytes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&inputBuffer, nInputBytes, MPI_BYTE, 0, MPI_COMM_WORLD);

    // Execute the actual service where results are stored in rawOutputData
    // This will be executed on all worker and operative threads
    manager->m[serviceID]->run(inputBuffer, inputBufferLength, outputBuffer, outputBufferLength);

    void * gatheredData;
    MPI_Gather(
            &rawOutputData,
            nOutputBytes,
            MPI_BYTE,
            &gatheredData,
            MPI_BYTE,
            0,
            MPI_COMM_WORLD
    );
}

void MPIMessaging::finalize() {
    MPI_Finalize();
}