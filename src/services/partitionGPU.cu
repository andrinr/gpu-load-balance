#include "partitionGPU.h"
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
        unsigned int * g_begins,
        unsigned int * g_ends,
        unsigned int * g_nLeft,
        unsigned int * g_axis,
        unsigned int * g_cellIndices,
        float * g_cuts,
        unsigned int * g_offsetLessEquals,
        unsigned int * g_offsetGreater,
        float * g_particlesX,
        float * g_particlesY,
        float * g_particlesZ,
        float * g_odata,
        unsigned int * g_permutations) {

    __shared__ unsigned int s_lessEquals[blockSize * 2];
    __shared__ unsigned int s_greater[blockSize * 2];

    __shared__ unsigned int s_offsetLessEquals;
    __shared__ unsigned int s_offsetGreater;

    unsigned int tid = threadIdx.x;
    const unsigned int begin = g_begins[blockIdx.x];
    const unsigned int end = g_ends[blockIdx.x];
    const float cut = g_cuts[blockIdx.x];
    const unsigned int cellIndex = g_cellIndices[blockIdx.x];
    const unsigned int nLeft = g_nLeft[blockIdx.x];
    const unsigned int axis = g_axis[blockIdx.x];

    unsigned int i = begin + 2 * tid;
    unsigned int j = begin + 2 * tid + 1;

    bool f1, f2;
    if (i < end) {
        if (axis == 0) {
            f1 = g_particlesX[i] <= cut;
            f2 = not f1;
        }
        else if (axis == 1) {
            f1 = g_particlesY[i] <= cut;
            f2 = not f1;
        }
        else {
            f1 = g_particlesZ[i] <= cut;
            f2 = not f1;
        }
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
    if (j < end) {
        if (axis == 0) {
            f3 = g_particlesX[j] <= cut;
            f4 = not f3;
        }
        else if (axis == 1) {
            f3 = g_particlesY[j] <= cut;
            f4 = not f3;
        }
        else {
            f3 = g_particlesZ[j] <= cut;
            f4 = not f3;
        }
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
        s_offsetLessEquals = atomicAdd(&g_offsetLessEquals[cellIndex], s_lessEquals[blockSize * 2 - 1] + f3);
        s_offsetGreater = atomicAdd(&g_offsetGreater[cellIndex], s_greater[blockSize * 2 - 1] + f4);
    }

    __syncthreads();

    // avoiding warp divergence
    unsigned int indexA = (s_lessEquals[2*tid] + s_offsetLessEquals) * f1 +
                          (s_greater[2*tid] + s_offsetGreater + nLeft) * f2;

    unsigned int indexB = (s_lessEquals[2*tid+1] + s_offsetLessEquals) * f3 +
                          (s_greater[2*tid+1] + s_offsetGreater + nLeft) * f4;

    if (i < end) {
        if (axis == 0) {
            g_odata[indexA] = g_particlesX[i];
        }
        else if (axis == 1) {
            g_odata[indexA] = g_particlesY[i];
        }
        else {
            g_odata[indexA] = g_particlesZ[i];
        }
        //g_odata[i] = cellIndex;
        g_permutations[i] = indexA;
    }

    if (j < end) {
        if (axis == 0) {
            g_odata[indexB] = g_particlesX[j];
        }
        else if (axis == 1) {
            g_odata[indexB] = g_particlesY[j];
        }
        else {
            g_odata[indexB] = g_particlesZ[j];
        }
        //g_odata[j] = nLeft;
        g_permutations[j] = indexB;
    }
}

template <unsigned int blockSize>
__global__ void permute(
        unsigned int * g_begins,
        unsigned int * g_ends,
        unsigned int * g_axis,
        float * g_particlesX,
        float * g_particlesY,
        float * g_particlesZ,
        float * g_odata,
        unsigned int * g_permutations,
        unsigned int rollAxis) {
    unsigned int tid = threadIdx.x;

    const unsigned int begin = g_begins[blockIdx.x];
    const unsigned int end = g_ends[blockIdx.x];
    const unsigned int axis = g_axis[blockIdx.x];

    unsigned int i = begin + 2 * tid;
    unsigned int j = begin + 2 * tid + 1;
    //unsigned int gridSize = blockSize*2*gridDim.x;

    if (i < end) {
        if (axis == rollAxis) {
            g_odata[g_permutations[i]] = g_particlesX[i];
        }
        else if (axis == rollAxis + 1) {
            g_odata[g_permutations[i]] = g_particlesY[i];
        }
        else {
            g_odata[g_permutations[i]] = g_particlesZ[i];
        }
        //g_odata[g_permutations[i]] = g_idata[i];
    }

    if (j < end) {
        if (axis == rollAxis) {
            g_odata[g_permutations[j]] = g_particlesX[j];
        }
        else if (axis == rollAxis + 1) {
            g_odata[g_permutations[j]] = g_particlesY[j];
        }
        else {
            g_odata[g_permutations[j]] = g_particlesZ[j];
        }
        //g_odata[g_permutations[j]] = g_idata[j];
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
            printf("%d\n", lcl->h_nLefts[blockPtr]);
            lcl->h_cellIndices[blockPtr] = cellPtrOffset;
            lcl->h_axis[blockPtr] = cell.cutAxis;
            lcl->h_cuts[blockPtr] = cell.getCut();
            lcl->h_begins[blockPtr] = begin;
            begin += N_THREADS * ELEMENTS_PER_THREAD;
            lcl->h_ends[blockPtr] = min(begin, endInd);

            blockPtr++;
        }
        nBlocks += nBlocksPerCell;

        out[cellPtrOffset] = 0;
    }

    CUDA_CHECK(cudaMemcpyAsync,(
            lcl->d_cellIndices,
            lcl->h_cellIndices,
            sizeof (unsigned int) * nBlocks,
            cudaMemcpyHostToDevice,
            lcl->streams(0)));

    CUDA_CHECK(cudaMemcpyAsync,(
            lcl->d_begins,
            lcl->h_begins,
            sizeof (unsigned int) * nBlocks,
            cudaMemcpyHostToDevice,
            lcl->streams(0)));

    CUDA_CHECK(cudaMemcpyAsync,(
            lcl->d_ends,
            lcl->h_ends,
            sizeof (unsigned int) * nBlocks,
            cudaMemcpyHostToDevice,
            lcl->streams(0)));

    CUDA_CHECK(cudaMemcpyAsync,(
            lcl->d_axis,
            lcl->h_axis,
            sizeof (unsigned int) * nBlocks,
            cudaMemcpyHostToDevice,
            lcl->streams(0)));

    CUDA_CHECK(cudaMemcpyAsync,(
            lcl->d_cuts,
            lcl->h_cuts,
            sizeof (float) * nBlocks,
            cudaMemcpyHostToDevice,
            lcl->streams(0)));

    CUDA_CHECK(cudaMemcpyAsync,(
            lcl->d_nLefts,
            lcl->h_nLefts,
            sizeof (float) * nBlocks,
            cudaMemcpyHostToDevice,
            lcl->streams(0)));

    CUDA_CHECK(cudaMemset, (lcl->d_offsetLeq, 0, sizeof(unsigned int) * nCells));
    CUDA_CHECK(cudaMemset, (lcl->d_offsetG, 0, sizeof(unsigned int) * nCells));

    partition<N_THREADS><<<
        nBlocks,
        N_THREADS,
        N_THREADS * sizeof(unsigned int) * 4 + sizeof(unsigned int) * 2,
        lcl->streams(0)
    >>>(
            lcl->d_begins,
            lcl->d_ends,
            lcl->d_nLefts,
            lcl->d_axis,
            lcl->d_cellIndices,
            lcl->d_cuts,
            lcl->d_offsetLeq,
            lcl->d_offsetG,
            lcl->d_particlesX,
            lcl->d_particlesY,
            lcl->d_particlesZ,
            lcl->d_particlesT,
            lcl->d_permutations
    );

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

    
    permute<N_THREADS><<<
        nBlocks,
        N_THREADS,
        0,
        lcl->streams(0)
    >>>(
            lcl->d_begins,
            lcl->d_ends,
            lcl->d_axis,
            lcl->d_particlesX,
            lcl->d_particlesY,
            lcl->d_particlesZ,
            lcl->d_particlesT,
            lcl->d_permutations,
            1
    );

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

    permute<N_THREADS><<<
        nBlocks,
        N_THREADS,
        0,
        lcl->streams(0)
    >>>(
            lcl->d_begins,
            lcl->d_ends,
            lcl->d_axis,
            lcl->d_particlesX,
            lcl->d_particlesY,
            lcl->d_particlesZ,
            lcl->d_particlesT,
            lcl->d_permutations,
            2
    );

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

    }

    return 0;
}

int ServicePartitionGPU::Combine(void *vout,void *vout2,int nIn,int nOut1,int nOut2) {
    return 0;
}
