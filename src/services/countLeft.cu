#include "countLeft.cuh"
#include <blitz/array.h>
#include <array>
#include "../utils/reduce.cuh"
// Make sure that the communication structure is "trivial" so that it
// can be moved around with "memcpy" which is required for MDL.
static_assert(std::is_void<ServiceCountLeftGPU::input>()  || std::is_trivial<ServiceCountLeftGPU::input>());
static_assert(std::is_void<ServiceCountLeftGPU::output>() || std::is_trivial<ServiceCountLeftGPU::output>());

int ServiceCountLeftGPU::Service(PST pst,void *vin,int nIn,void *vout, int nOut) {
    // store streams / initialize in local d
    // ata
    auto lcl = pst->lcl;
    auto in  = static_cast<input *>(vin);
    auto out = static_cast<output *>(vout);
    const int nCells = nIn / sizeof(input);
    assert(nOut / sizeof(output) >= nCells);

    //int bytes = nCounts * sizeof (uint);
    // https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/
    int blockOffset = 0;
    std::array<int, MAX_CELLS> offsets;
    offsets[0] = 0;

    for (int cellPtrOffset = 0; cellPtrOffset < nCells; ++cellPtrOffset) {
        out[cellPtrOffset] = 0;
    }

    for (int cellPtrOffset = 0; cellPtrOffset < nCells; ++cellPtrOffset) {
        auto cell = static_cast<Cell>(*(in + cellPtrOffset));

        if (cell.foundCut) {
            offsets[cellPtrOffset+1] = blockOffset;
            continue;
        }
        int beginInd = pst->lcl->cellToRangeMap(cell.id, 0);
        int endInd =  pst->lcl->cellToRangeMap(cell.id, 1);
        int n = endInd - beginInd;
        float cut = cell.getCut();

        if (n > 1 << 12) {
            const int nBlocks = (int) ceil((float) n / (N_THREADS * ELEMENTS_PER_THREAD));

            conditionalReduce<N_THREADS, true>(
                    lcl->d_particles + beginInd,
                    lcl->d_resultsA + blockOffset,
                    cut,
                    n,
                    nBlocks,
                    N_THREADS,
                    N_THREADS * sizeof (uint),
                    lcl->stream
                    );

            blockOffset += nBlocks;
        }
        else {
            blitz::Array<float,1> particles =
                    pst->lcl->particles(blitz::Range(beginInd, endInd), 0);

            float * startPtr = particles.data();
            float * endPtr = startPtr + (endInd - beginInd);

            for(auto p= startPtr; p<endPtr; ++p)
            {
                out[cellPtrOffset] += *p < cut;
            }
        }

        offsets[cellPtrOffset+1] = blockOffset;
    }

    CUDA_CHECK(cudaMemcpyAsync,(
            lcl->h_resultsA,
            lcl->d_resultsA,
            sizeof (uint) * blockOffset,
            cudaMemcpyDeviceToHost,
            lcl->stream));

    CUDA_CHECK(cudaStreamSynchronize,(lcl->stream));

    for (int cellPtrOffset = 0; cellPtrOffset < nCells; ++cellPtrOffset) {
        int begin = offsets[cellPtrOffset];
        int end = offsets[cellPtrOffset + 1];

        for (int i = begin; i < end; ++i) {
            out[cellPtrOffset] += lcl->h_resultsA[i];
        }
    }

    return nCells * sizeof(output);
}

int ServiceCountLeftGPU::Combine(void *vout,void *vout2,int nIn,int nOut1,int nOut2) {
    auto out  = static_cast<output *>(vout);
    auto out2 = static_cast<output *>(vout2);
    int nCounts = nIn / sizeof(input);
    assert(nOut1 >= nCounts*sizeof(output));
    assert(nOut2 >= nCounts*sizeof(output));
    for(auto i=0; i<nCounts; ++i)
	    out[i] += out2[i];
    return nCounts * sizeof(output);
}
