
template <unsigned int blockSize, bool leq>
void conditionalReduce(
        float *g_idata,
        uint *g_odata,
        float cut,
        uint n,
        uint nBlocks,
        uint nThreads,
        uint sharedMemBytes,
        cudaStream_t stream);

