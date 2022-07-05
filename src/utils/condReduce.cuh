
template <unsigned int blockSize, bool leq>
void conditionalReduce(
        float *g_idata,
        unsigned int*g_odata,
        float cut,
        int n,
        int nBlocks,
        int nThreads,
        int sharedMemBytes,
        cudaStream_t stream);
