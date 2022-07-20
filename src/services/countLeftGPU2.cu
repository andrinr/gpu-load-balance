#include "countLefGPU.h"
#include <blitz/array.h>
#include <array>
#include "../utils/condReduce.cuh"

// Make sure that the communication structure is "trivial" so that it
// can be moved around with "memcpy" which is required for MDL.
static_assert(std::is_void<ServiceCountLeftGPU::input>()  || std::is_trivial<ServiceCountLeftGPU::input>());
static_assert(std::is_void<ServiceCountLeftGPU::output>() || std::is_trivial<ServiceCountLeftGPU::output>());

#include "buildTreeGPU.h"
#include <blitz/array.h>
#include <vector>

// Make sure that the communication structure is "trivial" so that it
// can be moved around with "memcpy" which is required for MDL.
static_assert(std::is_void<ServiceCopyToDevice::input>()  || std::is_trivial<ServiceCopyToDevice::input>());
static_assert(std::is_void<ServiceCopyToDevice::output>() || std::is_trivial<ServiceCopyToDevice::output>());

template <unsigned int blockSize>
__device__ void warpReduce(volatile int *sdata, unsigned int tid, int nCells) {
    if (blockSize >= 64) {
        for (int i = 0; i < nCells; i++) {
            sdata[tid * nCells + i] += sdata[tid * nCells + i + 32];
        }
        //sdata[tid] += sdata[tid + 32];
    }
    if (blockSize >= 32) {
        for (int i = 0; i < nCells; i++) {
            sdata[tid * nCells + i] += sdata[tid * nCells + i + 16];
        }
        //sdata[tid] += sdata[tid + 16];
    }
    if (blockSize >= 16) {
        for (int i = 0; i < nCells; i++) {
            sdata[tid * nCells + i] += sdata[tid * nCells + i + 8];
        }
        //sdata[tid] += sdata[tid + 8];
    }
    if (blockSize >= 8) {
        for (int i = 0; i < nCells; i++) {
            sdata[tid * nCells + i] += sdata[tid * nCells + i + 4];
        }
        //sdata[tid] += sdata[tid + 4];
    }
    if (blockSize >= 4) {
        for (int i = 0; i < nCells; i++) {
            sdata[tid * nCells + i] += sdata[tid * nCells + i + 2];
        }
        //sdata[tid] += sdata[tid + 2];
    }
    if (blockSize >= 2) {
        for (int i = 0; i < nCells; i++) {
            sdata[tid * nCells + i] += sdata[tid * nCells + i + 1];
        }
        //sdata[tid] += sdata[tid + 1];
    }
}

template <unsigned int blockSize, unsigned int numCells>
__device__ void reduce(
        float * g_particles,
        unsigned int* g_cell,
        unsigned int* g_axis,
        float * g_cuts,
        unsigned int* g_counts,
        int n,
        int nCells)
{
    extern __shared__ int s_counts[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize) + threadIdx.x;
    unsigned int gridSize = blockSize * gridDim.x;

    while (i < n) {
        s_counts[tid * nCells + g_cell[i]] += (g_particles[i] < g_cuts[g_cell[i]]);
        i += gridSize;
    }
    __syncthreads();

    if (blockSize >= 512) {
        if (tid < 256) {
            for (int i = 0; i < nCells; i++) {
                s_counts[tid * nCells + i] += s_counts[(tid + 256) * nCells + i];
            }
            __syncthreads();
        }
    }
    if (blockSize >= 256) {
        if (tid < 128) {
            for (int i = 0; i < nCells; i++) {
                s_counts[tid * nCells + i] += s_counts[(tid + 128) * nCells + i];
            }
        }
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) {
            for (int i = 0; i < nCells; i++) {
                s_counts[tid * nCells + i] += s_counts[(tid + 64) * nCells + i];
            }
        }
        __syncthreads();
    }
    if (tid < 32) {
        warpReduce<blockSize>(sdata, tid, nCells);
    }
    if (tid == 0) {
        for (int i = 0; i < nCells; i++) {
            g_counts[i] = s_counts[i];
        }
    }
}

int ServiceBuildTreeGPU::Service(PST pst,void *vin,int nIn,void *vout, int nOut) {
    // store streams / initialize in local data
    auto lcl = pst->lcl;
    auto in  = static_cast<input *>(vin);
    auto out = static_cast<output *>(vout);
    const int nCells = nIn / sizeof(input);

    const int nBlocks = (int) ceil((float) lcl->particles.rows() / (N_THREADS * ELEMENTS_PER_THREAD));

    CUDA_CHECK(cudaMemcpyAsync,(
            lcl->h_cuts,
                    lcl->d_cuts,
                    sizeof (float ) * nCells,
                    cudaMemcpyHostToDevice,
                    lcl->stream));

    reduce<N_THREADS>
    <<<
    nBlocks,
    N_THREADS,
    N_THREADS * sizeof (uint) * nCells,
    lcl->stream
    >>>
            (lcl->d_particles,
             lcl->d_counts,
             lcl->d_cells,
             lcl->d_axis,
             lcl->d_cuts,
             lcl->g_counts,
             n,
             nCells);

    CUDA_CHECK(cudaMemcpyAsync,(
            lcl->h_counts,
                    lcl->d_counts,
                    sizeof (uint) * nBlocks,
                    cudaMemcpyDeviceToHost,
                    lcl->stream));

    CUDA_CHECK(cudaStreamSynchronize,(lcl->stream));

    for (int cellPtrOffset = 0; cellPtrOffset < nCells; ++cellPtrOffset) {
        for (int i = 0; i < nBlocks; ++i) {
            out[cellPtrOffset] += lcl->h_counts[i * nCells + cellPtrOffset];
        }
    }

    return sizeof(output);
}

int ServiceBuildTreeGPU::Combine(void *vout,void *vout2,int nIn,int nOut1,int nOut2) {
    auto out  = static_cast<output *>(vout);
    auto out2 = static_cast<output *>(vout2);
    int nCounts = nIn / sizeof(input);
    assert(nOut1 >= nCounts*sizeof(output));
    assert(nOut2 >= nCounts*sizeof(output));
    for(auto i=0; i<nCounts; ++i)
        out[i] += out2[i];
    return nCounts * sizeof(output);
}


template <unsigned int blockSize>
extern __device__ void warpReduce(volatile int *sdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize, bool leq>
extern __global__ void reduce(float *g_idata, unsigned int*g_odata, float cut, int n) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize) + threadIdx.x;
    unsigned int gridSize = blockSize*gridDim.x;
    sdata[tid] = 0;

    while (i < n) {
        if (leq){
            sdata[tid] += (g_idata[i] <= cut);
        } else {
            sdata[tid] += (g_idata[i] > cut);
        }
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
    unsigned intblockOffset = 0;
    std::array<uint, MAX_CELLS> offsets;
    offsets[0] = 0;

    for (int cellPtrOffset = 0; cellPtrOffset < nCells; ++cellPtrOffset) {
        out[cellPtrOffset] = 0;
    }

    // Make sure memcopy is done for this thread
    // Could also improved but seems complicated
    CUDA_CHECK(cudaStreamSynchronize,(lcl->streams(0)));
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
            const bool leq = true;

            // Kernel launches are serialzed within stream !
            // increase number of streams per thread

            reduce<N_THREADS, leq>
            <<<
                    nBlocks,
                    N_THREADS,
                    N_THREADS * sizeof (uint),
                    lcl->streams(cellPtrOffset % N_STREAMS)
            >>> (
                    lcl->d_particles + beginInd,
                    lcl->d_resultsA + blockOffset,
                    cut,
                    n
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

    for (int i = 0; i < N_STREAMS; i++) {
        CUDA_CHECK(cudaStreamSynchronize,(lcl->streams(i)));
    }

    CUDA_CHECK(cudaMemcpyAsync,(
            lcl->h_resultsA,
                    lcl->d_resultsA,
                    sizeof (uint) * blockOffset,
                    cudaMemcpyDeviceToHost,
                    lcl->streams(0)));

    CUDA_CHECK(cudaStreamSynchronize,(lcl->streams(0)));

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
