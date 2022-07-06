#include "countLeft.cuh"
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

template <unsigned int blockSize, bool leq>
extern __global__ void reduce(float *g_idata, unsigned int * g_odata, float cut, int n) {
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
        atomicAdd(g_odata, sdata[0]);
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


    // Make sure memcopy is done for this thread
    // Could also improved but seems complicated
    CUDA_CHECK(cudaMemset, (lcl->d_results, 0, nCells * sizeof(unsigned int)));
    CUDA_CHECK(cudaStreamSynchronize,(lcl->streams(0)));
    for (int cellPtrOffset = 0; cellPtrOffset < nCells; ++cellPtrOffset) {
        auto cell = static_cast<Cell>(*(in + cellPtrOffset));

        if (cell.foundCut) continue;

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
                    lcl->d_results + cellPtrOffset,
                    cut,
                    n
            );
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
    }

    for (int i = 0; i < N_STREAMS; i++) {
        CUDA_CHECK(cudaStreamSynchronize,(lcl->streams(i)));
    }

    CUDA_CHECK(cudaMemcpyAsync,(
            lcl->h_results,
            lcl->d_results,
            sizeof (uint) * nCells,
            cudaMemcpyDeviceToHost,
            lcl->streams(0)));

    CUDA_CHECK(cudaStreamSynchronize,(lcl->streams(0)));

    for (int cellPtrOffset = 0; cellPtrOffset < nCells; ++cellPtrOffset) {
        out[cellPtrOffset] = lcl->h_results[cellPtrOffset];
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
