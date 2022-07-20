#include "countLefGPU.h"
#include <blitz/array.h>
#include <array>
#include "../utils/condReduce.cuh"

// Make sure that the communication structure is "trivial" so that it
// can be moved around with "memcpy" which is required for MDL.
static_assert(std::is_void<ServiceCountLeftGPU::input>()  || std::is_trivial<ServiceCountLeftGPU::input>());
static_assert(std::is_void<ServiceCountLeftGPU::output>() || std::is_trivial<ServiceCountLeftGPU::output>());

template <unsigned int blockSize>
extern __device__ void warpReduce(volatile int *sdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
extern __global__ void reduce(float *g_idata, unsigned int*g_odata, float cut, int n) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize) + threadIdx.x;
    unsigned int gridSize = blockSize*gridDim.x;
    sdata[tid] = 0;

    while (i < n) {
        sdata[tid] += (g_idata[i] <= cut);
        i += gridSize;
    }
    __syncthreads();

    if (blockSize >= 512) {
        if (tid < 256) {
            sdata[tid] += sdata[tid + 256];
        }
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) {
            sdata[tid] += sdata[tid + 128];
        } __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) {
            sdata[tid] += sdata[tid + 64];
        } __syncthreads();
    }
    if (tid < 32) {
        warpReduce<blockSize>(sdata, tid);
    }
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

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
    unsigned int blockOffset = 0;
    std::array<unsigned int, MAX_CELLS> offsets;
    offsets[0] = 0;

    // Make sure memcopy is done for this thread
    // Could also improved but seems complicated

    for (int cellPtrOffset = 0; cellPtrOffset < nCells; ++cellPtrOffset) {
        auto cell = static_cast<Cell>(*(in + cellPtrOffset));

        if (cell.foundCut) {
            offsets[cellPtrOffset+1] = blockOffset;
            continue;
        }

        out[cellPtrOffset] = 0;
        int beginInd = pst->lcl->cellToRangeMap(cell.id, 0);
        int endInd =  pst->lcl->cellToRangeMap(cell.id, 1);
        int n = endInd - beginInd;
        float cut = cell.getCut();

        if (n > 1){ //N_THREADS * ELEMENTS_PER_THREAD) {
            const int nBlocks = (int) floor((float) n / (N_THREADS * ELEMENTS_PER_THREAD));
            const bool leq = true;

            // Kernel launches are serialzed within stream !
            // increase number of streams per thread
            if (cell.cutAxis == 0) {
                reduce<N_THREADS>
                <<<
                    nBlocks,
                    N_THREADS,
                    N_THREADS * sizeof (uint),
                    lcl->streams(0)
                >>> (
                    lcl->d_particlesX + beginInd,
                    lcl->d_results + blockOffset,
                    cut,
                    n
                );
            }
            else if (cell.cutAxis == 1) {
                reduce<N_THREADS>
                <<<
                    nBlocks,
                    N_THREADS,
                    N_THREADS * sizeof (uint),
                    lcl->streams(0)
                >>> (
                    lcl->d_particlesY + beginInd,
                    lcl->d_results + blockOffset,
                    cut,
                    n
                );
            }
            else if (cell.cutAxis == 2) {
                reduce<N_THREADS>
                <<<
                    nBlocks,
                    N_THREADS,
                    N_THREADS * sizeof (uint),
                    lcl->streams(0)
                >>> (
                    lcl->d_particlesZ + beginInd,
                    lcl->d_results + blockOffset,
                    cut,
                    n
                );
            }

            blockOffset += nBlocks;
        }
        else {
            blitz::Array<float,1> particles =
                    pst->lcl->particlesT(blitz::Range(beginInd, endInd), 0);

            float * startPtr = particles.data();
            float * endPtr = startPtr + (endInd - beginInd);

            for(auto p= startPtr; p<endPtr; ++p)
            {
                out[cellPtrOffset] += *p < cut;
            }
            lcl->h_countsLeft(cellPtrOffset) = out[cellPtrOffset];
        }

        offsets[cellPtrOffset+1] = blockOffset;
    }

    CUDA_CHECK(cudaMemcpyAsync,(
            lcl->h_results,
            lcl->d_results,
            sizeof (unsigned int) * blockOffset,
            cudaMemcpyDeviceToHost,
            lcl->streams(0)));

    CUDA_CHECK(cudaStreamSynchronize,(lcl->streams(0)));

    for (int cellPtrOffset = 0; cellPtrOffset < nCells; ++cellPtrOffset) {

        auto cell = static_cast<Cell>(*(in + cellPtrOffset));
        if (cell.foundCut) {continue;}

        int begin = offsets[cellPtrOffset];
        int end = offsets[cellPtrOffset + 1];

        for (int i = begin; i < end; ++i) {
            out[cellPtrOffset] += lcl->h_results[i];
        }

        lcl->h_countsLeft(cellPtrOffset) = out[cellPtrOffset];
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
