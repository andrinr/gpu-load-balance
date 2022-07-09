#include "partition.cuh"
#include "../cell.h"
#include <blitz/array.h>
#include "../utils/condReduce.cuh"

// Make sure that the communication structure is "trivial" so that it
// can be moved around with "memcpy" which is required for MDL.
static_assert(std::is_void<ServicePartitionGPU::input>()  || std::is_trivial<ServicePartitionGPU::input>());
static_assert(std::is_void<ServicePartitionGPU::output>() || std::is_trivial<ServicePartitionGPU::output>());

__device__ void scan(volatile unsigned int * s_idata, unsigned int thid, unsigned int n) {
    unsigned int offset = 1;
    for (unsigned int d = n>>1; d > 0; d >>= 1) // build sum in place up the tree
    {
        __syncthreads();
        if (thid < d)
        {
            unsigned int ai = offset*(2*thid+1)-1;
            unsigned int bi = offset*(2*thid+2)-1;
            s_idata[bi] += s_idata[ai];
        }
        offset *= 2;
    }
    if (thid == 0) { s_idata[n - 1] = 0; } // clear the last element
    for (unsigned int d = 1; d < n; d *= 2) // traverse down tree & build scan
    {
        offset >>= 1;
        __syncthreads();
        if (thid < d)
        {
            unsigned int ai = offset*(2*thid+1)-1;
            unsigned int bi = offset*(2*thid+2)-1;
            unsigned int t = s_idata[ai];
            s_idata[ai] = s_idata[bi];
            s_idata[bi] += t;
        }
    }
}

template <unsigned int blockSize>
__global__ void partition(
        unsigned int * g_offsetLessEquals,
        unsigned int * g_offsetGreater,
        float * g_idata,
        float * g_odata,
        unsigned int * g_permutations,
        float pivot,
        unsigned int nLeft,
        unsigned int n) {
    __shared__ unsigned int s_lessEquals[blockSize * 2];
    __shared__ unsigned int s_greater[blockSize * 2];

    __shared__ unsigned int s_offsetLessEquals;
    __shared__ unsigned int s_offsetGreater;

    unsigned int tid = threadIdx.x;

    unsigned int i = blockIdx.x * blockSize * 2 + 2 * tid;
    unsigned int j = blockIdx.x * blockSize * 2 + 2 * tid + 1;
    //unsigned int gridSize = blockSize*2*gridDim.x;

    bool f1, f2;
    if (i < n) {
        f1 = g_idata[i] <= pivot;
        f2 = not f1;
        // potential to avoid bank conflicts here
        s_lessEquals[2*tid] = f1;
        s_greater[2*tid] = f2;
    }
    else {
        f1 = false;
        f2 = false;
        s_lessEquals[2*tid] = 0;
        s_greater[2*tid] = 0;
    }

    bool f3, f4;
    if (j < n) {
        f3 = g_idata[j] <= pivot;
        f4 = not f3;
        // potential to avoid bank conflicts here
        s_lessEquals[2*tid+1] = f3;
        s_greater[2*tid+1] = f4;
    }
    else {
        f3 = false;
        f4 = false;
        s_lessEquals[2*tid+1] = 0;
        s_greater[2*tid+1] = 0;
    }

    __syncthreads();

    scan(s_lessEquals, tid, blockSize * 2 );
    scan(s_greater, tid, blockSize * 2);

    __syncthreads();

    // Avoid another kernel
    if (tid == blockSize - 1) {
        // result shared among kernel
        // atomicAdd returns old
        // exclusive scan does not include the last element
        s_offsetLessEquals = atomicAdd(g_offsetLessEquals, s_lessEquals[blockSize * 2 - 1] + f3);
        s_offsetGreater = atomicAdd(g_offsetGreater, s_greater[blockSize * 2 - 1] + f4);
    }

    __syncthreads();

    // avoiding warp divergence
    unsigned int indexA = (s_lessEquals[2*tid] + s_offsetLessEquals) * f1 +
                          (s_greater[2*tid] + s_offsetGreater + nLeft) * f2;

    unsigned int indexB = (s_lessEquals[2*tid+1] + s_offsetLessEquals) * f3 +
                          (s_greater[2*tid+1] + s_offsetGreater + nLeft) * f4;

    if (i < n) {
        g_odata[indexA] = g_idata[i];
        g_permutations[i] = indexA;
    }

    if (j < n) {
        g_odata[indexB] = g_odata[j];
        g_permutations[j] = indexB;
    }
}

template <unsigned int blockSize>
__global__ void permute(
        float * g_idata,
        float * g_odata,
        unsigned int * g_permutations,
        int n) {
    unsigned int tid = threadIdx.x;

    unsigned int i = blockIdx.x * blockSize * 2 + 2 * tid;
    unsigned int j = blockIdx.x * blockSize * 2 + 2 * tid + 1;
    //unsigned int gridSize = blockSize*2*gridDim.x;

    if (i < n) {
        g_odata[g_permutations[i]] = g_idata[i];
    }

    if (j < n) {
        g_odata[g_permutations[j]] = g_odata[j];
    }
}

template <unsigned int blockSize>
__global__ void copy(
        float * g_idata,
        float * g_odata,
        unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockSize * 2 + 2 * tid;
    unsigned int j = blockIdx.x * blockSize * 2 + 2 * tid + 1;

    if (i < n) {
        g_odata[i] = g_idata[i];
    }
    if (j < n) {
        g_odata[j] = g_idata[j];
    }
}

int ServicePartitionGPU::Service(PST pst,void *vin,int nIn,void *vout, int nOut) {

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

    int countLeq[nCells];
    int countG[nCells];

    for (int cellPtrOffset = 0; cellPtrOffset < nCells; ++cellPtrOffset) {
        out[cellPtrOffset] = 0;
    }

    for (int cellPtrOffset = 0; cellPtrOffset < nCells; ++cellPtrOffset) {
        auto cell = static_cast<Cell>(*(in + cellPtrOffset));

        int beginInd = pst->lcl->cellToRangeMap(cell.id, 0);
        int endInd =  pst->lcl->cellToRangeMap(cell.id, 1);
        int n = endInd - beginInd;
        float cut = cell.getCut();

        const int nBlocks = (int) ceil((float) n / (N_THREADS *  ELEMENTS_PER_THREAD));

        unsigned int * d_offsetLessEquals;
        unsigned int * d_offsetGreater;

        CUDA_CHECK(cudaMalloc,(&d_offsetLessEquals, sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc,(&d_offsetGreater, sizeof(unsigned int)));

        CUDA_CHECK(cudaMemset,(d_offsetLessEquals, 0, sizeof (unsigned int)));
        CUDA_CHECK(cudaMemset,(d_offsetGreater, 0, sizeof (unsigned int)));

        // Case axis X
        if (cell.cutAxis == 0) {
            partition<N_THREADS><<<
                nBlocks,
                N_THREADS,
                N_THREADS * sizeof (uint) * 4 + sizeof (unsigned int) * 2
            >>>(
                d_offsetLessEquals,
                d_offsetGreater,
                lcl->d_particlesX + beginInd,
                lcl->d_particlesT + beginInd,
                lcl->d_permutations + beginInd,
                cell.getCut(),
                lcl->h_countsLeft(cellPtrOffset),
                n
            );

            copy<N_THREADS><<<
                nBlocks,
                N_THREADS
            >>>(
                    lcl->d_particlesT + beginInd,
                    lcl->d_particlesX + beginInd,
                    n
            );

            permute<N_THREADS><<<
                nBlocks,
                N_THREADS
            >>>(
                lcl->d_particlesY + beginInd,
                lcl->d_particlesT + beginInd,
                lcl->d_permutations + beginInd,
                n
            );

            copy<N_THREADS><<<
                nBlocks,
                N_THREADS
            >>>(
                    lcl->d_particlesT + beginInd,
                    lcl->d_particlesY + beginInd,
                    n
            );

            permute<N_THREADS><<<
                nBlocks,
                N_THREADS
            >>>(
                lcl->d_particlesZ + beginInd,
                lcl->d_particlesT + beginInd,
                lcl->d_permutations + beginInd,
                n
            );

            copy<N_THREADS><<<
                nBlocks,
                N_THREADS
            >>>(
                lcl->d_particlesT + beginInd,
                lcl->d_particlesZ + beginInd,
                n
            );
        }

        // case axis Y
        if (cell.cutAxis == 1) {
            partition<N_THREADS><<<
                nBlocks,
                N_THREADS,
                N_THREADS * sizeof (uint) * 4 + sizeof (unsigned int) * 2
            >>>(
                d_offsetLessEquals,
                d_offsetGreater,
                lcl->d_particlesY + beginInd,
                lcl->d_particlesT + beginInd,
                lcl->d_permutations + beginInd,
                cell.getCut(),
                lcl->h_countsLeft(cellPtrOffset),
                n
            );

            copy<N_THREADS><<<
                nBlocks,
                N_THREADS
            >>>(
                lcl->d_particlesT + beginInd,
                lcl->d_particlesY + beginInd,
                n
            );

            permute<N_THREADS><<<
                nBlocks,
                N_THREADS
            >>>(
                lcl->d_particlesX + beginInd,
                lcl->d_particlesT + beginInd,
                lcl->d_permutations + beginInd,
                n
            );

            copy<N_THREADS><<<
                nBlocks,
                N_THREADS
            >>>(
                    lcl->d_particlesT + beginInd,
                    lcl->d_particlesX + beginInd,
                    n
            );

            permute<N_THREADS><<<
                nBlocks,
                N_THREADS
            >>>(
                lcl->d_particlesZ + beginInd,
                lcl->d_particlesT + beginInd,
                lcl->d_permutations + beginInd,
                n
            );

            copy<N_THREADS><<<
                nBlocks,
                N_THREADS
            >>>(
                    lcl->d_particlesT + beginInd,
                    lcl->d_particlesZ + beginInd,
                    n
            );
        }

        // case axis Z
        if (cell.cutAxis == 2) {
            partition<N_THREADS><<<
                nBlocks,
                N_THREADS,
                N_THREADS * sizeof (uint) * 4 + sizeof (unsigned int) * 2
            >>>(
                d_offsetLessEquals,
                d_offsetGreater,
                lcl->d_particlesZ + beginInd,
                lcl->d_particlesT + beginInd,
                lcl->d_permutations + beginInd,
                cell.getCut(),
                lcl->h_countsLeft(cellPtrOffset),
                n
            );

            copy<N_THREADS><<<
                nBlocks,
                N_THREADS
            >>>(
                    lcl->d_particlesT + beginInd,
                    lcl->d_particlesZ + beginInd,
                    n
            );

            permute<N_THREADS><<<
                nBlocks,
                N_THREADS
            >>>(
                lcl->d_particlesX + beginInd,
                lcl->d_particlesT + beginInd,
                lcl->d_permutations + beginInd,
                n
            );

            copy<N_THREADS><<<
                nBlocks,
                N_THREADS
            >>>(
                lcl->d_particlesT + beginInd,
                lcl->d_particlesX + beginInd,
                n
            );

            permute<N_THREADS><<<
                nBlocks,
                N_THREADS
            >>>(
                lcl->d_particlesY + beginInd,
                lcl->d_particlesT + beginInd,
                lcl->d_permutations + beginInd,
                n
            );

            copy<N_THREADS><<<
                nBlocks,
                N_THREADS
            >>>(
                lcl->d_particlesT + beginInd,
                lcl->d_particlesY + beginInd,
                n
            );
        }

        printf("%d\n", lcl->h_countsLeft(cellPtrOffset));
        lcl->cellToRangeMap(cell.getLeftChildId(), 0) =
                lcl->cellToRangeMap(cell.id, 0);
        lcl->cellToRangeMap(cell.getLeftChildId(), 1) = lcl->h_countsLeft(cellPtrOffset);

        lcl->cellToRangeMap(cell.getRightChildId(), 0) = lcl->h_countsLeft(cellPtrOffset);
        lcl->cellToRangeMap(cell.getRightChildId(), 1) =
                lcl->cellToRangeMap(cell.id, 1);

        cudaFree(d_offsetGreater);
        cudaFree(d_offsetLessEquals);
    }

    return 0;
}

int ServicePartitionGPU::Combine(void *vout,void *vout2,int nIn,int nOut1,int nOut2) {
    return 0;
}
