#include "partitionGPU.h"
#include "../cell.h"
#include <blitz/array.h>
#include "../utils/condReduce.cuh"

// Make sure that the communication structure is "trivial" so that it
// can be moved around with "memcpy" which is required for MDL.
static_assert(std::is_void<ServicePartitionGPU::input>()  || std::is_trivial<ServicePartitionGPU::input>());
static_assert(std::is_void<ServicePartitionGPU::output>() || std::is_trivial<ServicePartitionGPU::output>());

#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(n) \
((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#endif

/*
__device__ void scan2(volatile unsigned int * s_idata, unsigned int thid, unsigned int n) {
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
__global__ void partition2(
        unsigned int * g_offsetLessEquals,
        unsigned int * g_offsetGreater,
        float * g_idata,
        float * g_odata,
        unsigned int * g_permutations,
        float pivot,
        unsigned int nLeft,
        unsigned int n) {

    int ai = thid;
    int bi = thid + (blockSize);
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(ai);

    __shared__ unsigned int s_lessEquals[blockSize * 2];
    __shared__ unsigned int s_greater[blockSize * 2];

    __shared__ unsigned int s_offsetLessEquals;
    __shared__ unsigned int s_offsetGreater;

    unsigned int tid = threadIdx.x;
    unsigned int l = blockIdx.x * blockSize * 2;

    unsigned int i = blockIdx.x * blockSize * 2 + 2 * tid;
    unsigned int j = blockIdx.x * blockSize * 2 + 2 * tid + 1;

    temp[ai + bankOffsetA] = g_idata[ai];
    temp[bi + bankOffsetB] = g_idata[bi];
    //unsigned int gridSize = blockSize*2*gridDim.x;

    temp[ai + bankOffsetA] = g_idata[ai + ];
    temp[bi + bankOffsetB] = g_idata[bi];
    bool f1, f2;
    if (i < n) {
        f1 = g_idata[ai + l] <= pivot;
        f2 = not f1;
        // potential to avoid bank conflicts here
        s_lessEquals[ai + bankOffsetA] = f1;
        s_greater[ai + bankOffsetA] = f2;
    }
    else {
        f1 = false;
        f2 = false;
        s_lessEquals[ai + bankOffsetA] = 0;
        s_greater[ai + bankOffsetA] = 0;
    }

    bool f3, f4;
    if (j < n) {
        f3 = g_idata[bi + left] <= pivot;
        f4 = not f3;
        // potential to avoid bank conflicts here
        s_lessEquals[bi + bankOffsetB] = f3;
        s_greater[bi + bankOffsetB] = f4;
    }
    else {
        f3 = false;
        f4 = false;
        s_lessEquals[bi + bankOffsetB] = 0;
        s_greater[bi + bankOffsetB] = 0;
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
        g_odata[indexB] = g_idata[j];
        g_permutations[j] = indexB;
    }
}
*/
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
        unsigned int * g_begins,
        unsigned int * g_ends,
        unsigned int * g_nLeft,
        float * g_cuts,
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
        g_odata[indexB] = g_idata[j];
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
        g_odata[g_permutations[j]] = g_idata[j];
    }
}

int ServicePartitionGPU::Service(PST pst,void *vin,int nIn,void *vout, int nOut) {

    auto lcl = pst->lcl;
    auto in  = static_cast<input *>(vin);
    auto out = static_cast<output *>(vout);
    const int nCells = nIn / sizeof(input);

    //int bytes = nCounts * sizeof (uint);
    // https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/
    int blockOffset = 0;
    std::array<int, MAX_CELLS> offsets;
    offsets[0] = 0;

    int countLeq[nCells];
    int countG[nCells];

    std::vector<unsigned  int> cellIndices;

    CUDA_CHECK(cudaMemset, (lcl->d_offsetLeq, 0, sizeof(unsigned int) * nCells));
    CUDA_CHECK(cudaMemset, (lcl->d_offsetG, 0, sizeof(unsigned int) * nCells));

    int nBlocks = 0;
    int blockPtr = 0;
    for (int cellPtrOffset = 0; cellPtrOffset < nCells; ++cellPtrOffset) {
        auto cell = static_cast<Cell>(*(in + cellPtrOffset));
        unsigned int beginInd = pst->lcl->cellToRangeMap(cell.id, 0);
        unsigned int endInd =  pst->lcl->cellToRangeMap(cell.id, 1);
        unsigned int n = endInd - beginInd;

        unsigned int nBlocksPerCell = (int) ceil((float) n / (N_THREADS * ELEMENTS_PER_THREAD));

        int begin = beginInd;
        for (int i = 0; i < nBlocksPerCell; ++i) {
            lcl->h_nLefts[blockPtr] = lcl->h_countsLeft(cellPtrOffset);
            cellIndices.push_back(cellPtrOffset);
            blockPtr++;
        }
        nBlocks += nBlocksPerCell;

        out[cellPtrOffset] = 0;
    }

    /*
    // Primary axis to temporary
    for (int cellPtrOffset = 0; cellPtrOffset < nCells; ++cellPtrOffset) {
        out[cellPtrOffset] = 0;

        auto cell = static_cast<Cell>(*(in + cellPtrOffset));

        int beginInd = pst->lcl->cellToRangeMap(cell.id, 0);
        int endInd = pst->lcl->cellToRangeMap(cell.id, 1);
        int n = endInd - beginInd;
        float cut = cell.getCut();

        const int nBlocks = (int) ceil((float) n / (N_THREADS * 2.0));

        //CUDA_CHECK(cudaMalloc, (&d_offsetLessEquals, sizeof(unsigned int)));
        //CUDA_CHECK(cudaMalloc, (&d_offsetGreater, sizeof(unsigned int)));
        float * d_from;
        float * d_to = lcl->d_particlesT + beginInd;

        if (cell.cutAxis == 0) {
            d_from = lcl->d_particlesX + beginInd;
        }
        else if (cell.cutAxis == 1) {
            d_from = lcl->d_particlesY + beginInd;
        }
        else {
            d_from = lcl->d_particlesZ + beginInd;
        }

        partition<N_THREADS><<<
            nBlocks,
            N_THREADS,
            N_THREADS * sizeof(unsigned int) * 4 + sizeof(unsigned int) * 2,
            lcl->streams(0)
        >>>(
                lcl->d_offsetLeq + cellPtrOffset,
                lcl->d_offsetG + cellPtrOffset,
                d_from,
                d_to,
                lcl->d_permutations + beginInd,
                cell.getCut(),
                lcl->h_countsLeft(cellPtrOffset),
                n
        );
    };

    // Temporary back to primary
    for (int cellPtrOffset = 0; cellPtrOffset < nCells; ++cellPtrOffset) {
        auto cell = static_cast<Cell>(*(in + cellPtrOffset));

        int beginInd = pst->lcl->cellToRangeMap(cell.id, 0);
        int endInd = pst->lcl->cellToRangeMap(cell.id, 1);
        int n = endInd - beginInd;

        float * d_from = lcl->d_particlesT + beginInd;
        float * d_to;

        if (cell.cutAxis == 0) {
            d_to = lcl->d_particlesX + beginInd;
        }
        else if (cell.cutAxis == 1) {
            d_to = lcl->d_particlesY + beginInd;
        }
        else {
            d_to = lcl->d_particlesZ + beginInd;
        }

        cudaMemcpyAsync(d_to, d_from, sizeof(float) * n, cudaMemcpyDeviceToDevice, lcl->streams(0));

    }

    // Secondary to temporary
    for (int cellPtrOffset = 0; cellPtrOffset < nCells; ++cellPtrOffset) {
        auto cell = static_cast<Cell>(*(in + cellPtrOffset));

        int beginInd = pst->lcl->cellToRangeMap(cell.id, 0);
        int endInd = pst->lcl->cellToRangeMap(cell.id, 1);
        int n = endInd - beginInd;

        const int nBlocks = (int) ceil((float) n / (N_THREADS * 2.0));

        float * d_from;
        float * d_to = lcl->d_particlesT + beginInd;

        if (cell.cutAxis == 0) {
            d_from = lcl->d_particlesY + beginInd;
        }
        else if (cell.cutAxis == 1) {
            d_from = lcl->d_particlesX + beginInd;
        }
        else {
            d_from = lcl->d_particlesX + beginInd;
        }

        permute<N_THREADS><<<
            nBlocks,
            N_THREADS,
            0,
            lcl->streams(0)
        >>>(
            d_from,
            d_to,
            lcl->d_permutations + beginInd,
            n
        );

    }

    // Temporary back to secondary
    for (int cellPtrOffset = 0; cellPtrOffset < nCells; ++cellPtrOffset) {
        auto cell = static_cast<Cell>(*(in + cellPtrOffset));

        int beginInd = pst->lcl->cellToRangeMap(cell.id, 0);
        int endInd = pst->lcl->cellToRangeMap(cell.id, 1);
        int n = endInd - beginInd;

        float * d_from = lcl->d_particlesT + beginInd;
        float * d_to;

        if (cell.cutAxis == 0) {
            d_to = lcl->d_particlesY + beginInd;
        }
        else if (cell.cutAxis == 1) {
            d_to = lcl->d_particlesX + beginInd;
        }
        else {
            d_to = lcl->d_particlesX + beginInd;
        }

        cudaMemcpyAsync(d_to, d_from,  sizeof(float) * n, cudaMemcpyDeviceToDevice, lcl->streams(0));
    }

    // Tertiary to temporary
    for (int cellPtrOffset = 0; cellPtrOffset < nCells; ++cellPtrOffset) {
        auto cell = static_cast<Cell>(*(in + cellPtrOffset));

        int beginInd = pst->lcl->cellToRangeMap(cell.id, 0);
        int endInd = pst->lcl->cellToRangeMap(cell.id, 1);
        int n = endInd - beginInd;

        const int nBlocks = (int) ceil((float) n / (N_THREADS * 2.0));

        float * d_from;
        float * d_to = lcl->d_particlesT + beginInd;

        if (cell.cutAxis == 0) {
            d_from = lcl->d_particlesZ + beginInd;
        }
        else if (cell.cutAxis == 1) {
            d_from = lcl->d_particlesZ + beginInd;
        }
        else {
            d_from = lcl->d_particlesY + beginInd;
        }

        permute<N_THREADS><<<
        nBlocks,
        N_THREADS,
        0,
        lcl->streams(0)
        >>>(
                d_from,
                d_to,
                lcl->d_permutations + beginInd,
                n
        );
    }

    // Temporary back to tertiary
    for (int cellPtrOffset = 0; cellPtrOffset < nCells; ++cellPtrOffset) {
        auto cell = static_cast<Cell>(*(in + cellPtrOffset));

        int beginInd = pst->lcl->cellToRangeMap(cell.id, 0);
        int endInd = pst->lcl->cellToRangeMap(cell.id, 1);
        int n = endInd - beginInd;

        float * d_from = lcl->d_particlesT + beginInd;
        float * d_to;

        if (cell.cutAxis == 0) {
            d_to = lcl->d_particlesZ + beginInd;
        }
        else if (cell.cutAxis == 1) {
            d_to = lcl->d_particlesZ + beginInd;
        }
        else {
            d_to = lcl->d_particlesY + beginInd;
        }

        cudaMemcpyAsync(d_to, d_from, sizeof(float) * n, cudaMemcpyDeviceToDevice, lcl->streams(0));

        lcl->cellToRangeMap(cell.getLeftChildId(), 0) =
                lcl->cellToRangeMap(cell.id, 0);
        lcl->cellToRangeMap(cell.getLeftChildId(), 1) =
                lcl->h_countsLeft(cellPtrOffset) + beginInd;

        lcl->cellToRangeMap(cell.getRightChildId(), 0) =
                lcl->h_countsLeft(cellPtrOffset) + beginInd;
        lcl->cellToRangeMap(cell.getRightChildId(), 1) =
                lcl->cellToRangeMap(cell.id, 1);
    }

    /*
    blitz::Array<float, 1> x = lcl->particles(blitz::Range::all(), 0);
    blitz::Array<float, 1> y = lcl->particles(blitz::Range::all(), 1);
    blitz::Array<float, 1> z = lcl->particles(blitz::Range::all(), 2);

    cudaMemcpyAsync(
            x.data(),
            lcl->d_particlesX,
            sizeof (float) * x.rows(),
            cudaMemcpyDeviceToHost,
            pst->lcl->streams(0)
    );

    cudaMemcpyAsync(
            y.data(),
            lcl->d_particlesY,
            sizeof (float) * y.rows(),
            cudaMemcpyDeviceToHost,
            pst->lcl->streams(0)
    );

    cudaMemcpyAsync(
            z.data(),
            lcl->d_particlesZ,
            sizeof (float) * z.rows(),
            cudaMemcpyDeviceToHost,
            pst->lcl->streams(0)
    );

    CUDA_CHECK(cudaStreamSynchronize,(lcl->streams(0)));

    if (pst->idSelf == 0) {
        for (int i = 0; i < x.rows(); i++) {
            printf("x %f y %f z %f  \n", x(i), y(i), z(i));
        }
        printf("\n---------------\n\n");

    }*/

    return 0;
}

int ServicePartitionGPU::Combine(void *vout,void *vout2,int nIn,int nOut1,int nOut2) {
    return 0;
}
