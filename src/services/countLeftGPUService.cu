#include "countLeftGPUService.h"
#include <blitz/array.h>
#include <vector>

// Make sure that the communication structure is "trivial" so that it
// can be moved around with "memcpy" which is required for MDL.
static_assert(std::is_void<ServiceCountLeftGPU::input>()  || std::is_trivial<ServiceCountLeftGPU::input>());
static_assert(std::is_void<ServiceCountLeftGPU::output>() || std::is_trivial<ServiceCountLeftGPU::output>());

template <unsigned int blockSize>
__device__ void warpReduce(volatile int *sdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void reduce(float *g_idata, uint *g_odata, float cut, int n) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = 0;
    while (i < n) {
        sdata[tid] += (g_idata[i] < cut);

        if (i + blockSize < n) {
            sdata[tid] += (g_idata[i+blockSize] < cut);
        }
        i += gridSize;
    }
    __syncthreads();

    if (blockSize >= 512) {
        if (tid < 256) {
            sdata[tid] += sdata[tid + 256];
        } __syncthreads();
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
    // store streams / initialize in local data
    auto lcl = pst->lcl;
    auto in  = static_cast<input *>(vin);
    auto out = static_cast<output *>(vout);
    auto nCells = nIn / sizeof(input);
    assert(nOut / sizeof(output) >= nCells);

    // https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/
    int blockOffset = 0;
    std::vector<int> offsets;
    for (int cellPtrOffset = 0; cellPtrOffset < nCells; ++cellPtrOffset) {
        auto cell = static_cast<Cell>(*(in + cellPtrOffset));

        if (cell.foundCut) {
            continue;
        }
        int beginInd = pst->lcl->cellToRangeMap(cell.id, 0);
        int endInd =  pst->lcl->cellToRangeMap(cell.id, 1);
        int n = endInd - beginInd;
        float cut = cell.getCut();

        out[cellPtrOffset] = 0;

        if (n > 1 << 12) {
            // Can increase speed by another factor of around two
            const int nBlocks = (int) ceil((float) n / (N_THREADS * 2.0 * ELEMENTS_PER_THREAD));

            uint * h_counts = (uint*)calloc(nBlocks, sizeof(int));

            printf("off %u blocks %u . \n", blockOffset, nBlocks);

            reduce<N_THREADS>
            <<<
                nBlocks,
                N_THREADS,
                N_THREADS * sizeof (uint),
                lcl->stream
            >>>
                    (lcl->d_particles + beginInd,
                     lcl->d_counts + blockOffset,
                     cut,
                     n - 5);

            offsets.push_back(blockOffset);
            blockOffset += nBlocks;

            //cudaStreamSynchronize(lcl->stream);

            for (int i = 0; i < nBlocks; ++i) {
                out[cellPtrOffset] += h_counts[i];
            }

            free(h_counts);
        }

        else {
            blitz::Array<float,1> particles =
                    pst->lcl->particles(blitz::Range(beginInd, endInd), 0);

            float * startPtr = particles.data();
            float * endPtr = startPtr + (endInd - beginInd);

            offsets.push_back(-1);

            for(auto p= startPtr; p<endPtr; ++p)
            {
                out[cellPtrOffset] += *p < cut;
            }
        }
    }

    CUDA_CHECK(cudaMemcpyAsync,(
            h_counts,
                    lcl->d_counts,
                    sizeof (uint) * blockOffset,
                    cudaMemcpyDeviceToHost,
                    lcl->stream));

    for (int cellPtrOffset = 0; cellPtrOffset < nCells; ++cellPtrOffset) {
        auto cell = static_cast<Cell>(*(in + cellPtrOffset));

        

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
