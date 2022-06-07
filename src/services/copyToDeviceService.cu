#include "copyToDeviceService.h"
#include <blitz/array.h>
#inlcude <vector>

// Make sure that the communication structure is "trivial" so that it
// can be moved around with "memcpy" which is required for MDL.
static_assert(std::is_void<ServiceCountLeft::input>()  || std::is_trivial<ServiceCountLeft::input>());
static_assert(std::is_void<ServiceCountLeft::output>() || std::is_trivial<ServiceCountLeft::output>());

int ServiceCopyToDevice::Service(PST pst,void *vin,int nIn,void *vout, int nOut) {
    // store streams / initialize in local data
    //
    auto lcl = pst->lcl;
    auto in  = static_cast<input *>(vin);
    auto out = static_cast<output *>(vout);
    auto nCells = nIn / sizeof(input);
    assert(nOut / sizeof(output) >= nCells);
    printf("ServiceCopyToDevice invoked on thread %d\n",pst->idSelf);

    cudaMalloc(&d_particles, sizeof (float) * n);
    cudaMemcpy(d_particles, pos, sizeof (float ) * n, cudaMemcpyHostToDevice);

    cudaMalloc(&d_sums, sizeof (int) * n);

    std::vector<cudaStream_t> streams;
    // https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/
    for (int cellPtrOffset = 0; cellPtrOffset < nCells; ++cellPtrOffset){

        auto cell = static_cast<Cell>*(in + cellPtrOffset);
        // -1 axis signals no need to count
        if (cell.cutAxis == -1) continue;
        int beginInd = pst->lcl->cellToRangeMap(cell.id, 0);
        int endInd =  pst->lcl->cellToRangeMap(cell.id, 1);

        const int nThreads = 256;
        // Can increase speed by another factor of around two
        const int elementsPerThread = 16;
        const int nBlocks = ceil(endInd - beginInd / (nThreads * elementsPerThread) / 2 );

        cudaStream_t stream;
        cudaError_t result;
        streams.push_back(stream);
        result = cudaStreamCreate(&stream);

        float * d_particles;
        int * d_counts;
        int * h_counts = (int*)calloc(nBlocks, sizeof(int));

        blitz::Array<float,1> particles = pst->lcl->particles(blitz::Range(beginInd, endInd), cell.cutAxis);

        cudaMalloc(&d_particles, sizeof (float) * n);
        cudaMalloc(&d_counts, sizeof (int) * nBlocks);
        result = cudaMemcpyAsync(d_particles, particles.data(), endInd - beginInd, cudaMemcpyHostToDevice, stream);

        float cut = (cell.cutMarginRight + cell.cutMarginLeft) / 2.0;
        reduce<nThreads><<<nBlocks, nThreads, nThreads * sizeof (int), stream>>>(
                d_particles, d_counts, cut, endInd - beginInd);

        cudaMemcpyAsync(h_sums, d_sums, sizeof (int ) * nBlocks, cudaMemcpyDeviceToHost, stream);

        for (int i = 0; i < nBlocks; ++i) {
            out[cellPtrOffset] += h_sums[i];
        }
    }

    // Wait till all streams have finished
    cudaDeviceSynchronize();

    for (auto stream : streams) {
        cudaStreamDestroy(stream);
    }

    return nCells * sizeof(output);
}

int ServiceCopyToDevice::Combine(void *vout,void *vout2,int nIn,int nOut1,int nOut2) {
    auto out  = static_cast<output *>(vout);
    auto out2 = static_cast<output *>(vout2);
    int nCounts = nIn / sizeof(input);
    assert(nOut1 >= nCounts*sizeof(output));
    assert(nOut2 >= nCounts*sizeof(output));
    for(auto i=0; i<nCounts; ++i)
	    out[i] += out2[i];
    return nCounts * sizeof(output);
}
