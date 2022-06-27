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

template <unsigned int blockSize>
__device__ void reduce(
        float * g_particles,
        uint * g_cell,
        uint * g_axis,
        float * g_cuts,
        uint * g_counts,
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
            //s_counts[tid] += s_counts[tid + 256];
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
           // g_counts[blockIdx.x] = s_counts[0];
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
