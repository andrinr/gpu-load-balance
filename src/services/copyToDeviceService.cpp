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

    const int nThreads = 256;
    // Can increase speed by another factor of around two
    const int elementsPerThread = 16;

    // https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/
    for (int cellPtrOffset = 0; cellPtrOffset < nCells; ++cellPtrOffset){

        auto cell = static_cast<Cell>*(in + cellPtrOffset);
        // -1 axis signals no need to count
        if (cell.cutAxis == -1) continue;
        int beginInd = pst->lcl->cellToRangeMap(cell.id, 0);
        int endInd =  pst->lcl->cellToRangeMap(cell.id, 1);

        const int nBlocks = ceil(endInd - beginInd / (nThreads * elementsPerThread) / 2 );

        int streamId = cellPtrOffset % 32;
        //cudaStreamSynchronize(lcl->streams(streamId));

        float * d_particles;
        pst->lcl->d_particles(i) = d_particles;

        int * d_counts;
        pst->lcl->d_counts(i) = d_counts;

        blitz::Array<float,1> particles = pst->lcl->particles(blitz::Range(beginInd, endInd), cell.cutAxis);

        cudaMalloc(&lcl->d_particles(cellPtrOffset), sizeof (float) * n);
        cudaMalloc(&lcl->d_counts(cellPtrOffset), sizeof (int) * nBlocks);
        cudaMemcpyAsync(
                d_particles,
                particles.data(),
                endInd - beginInd,
                cudaMemcpyHostToDevice,
                pst->lcl->streams(streamId)
        );
    }

    // Wait till all streams have finished
    cudaDeviceSynchronize();

    return nCells * sizeof(output);
}

int ServiceCopyToDevice::Combine(void *vout,void *vout2,int nIn,int nOut1,int nOut2) {

    return 0;
}
