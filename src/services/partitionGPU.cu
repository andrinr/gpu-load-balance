#include "partitionGPU.h"
#include "../cell.h"
#include <blitz/array.h>
#include <stdio.h>

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

__device__ void scan(volatile unsigned int * s_idata, unsigned int thid, unsigned int n, bool slow) {
    unsigned int offset = 1;
    for (unsigned int d = n>>1; d > 0; d >>= 1) // build sum in place up the tree
    {
        __syncthreads();
        if (thid < d)
        {
            unsigned int ai = offset*(2*thid+1)-1;
            unsigned int bi = offset*(2*thid+2)-1;
            if (not slow) {
                ai += CONFLICT_FREE_OFFSET(ai);
                bi += CONFLICT_FREE_OFFSET(bi);
            }
            s_idata[bi] += s_idata[ai];
        }
        offset *= 2;
    }
    if (not slow and thid==0) { s_idata[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0; } // clear the last element
    if (slow and thid==0) { s_idata[n - 1] = 0; } // clear the last element
    for (unsigned int d = 1; d < n; d *= 2) // traverse down tree & build scan
    {
        offset >>= 1;
        __syncthreads();
        if (thid < d)
        {
            unsigned int ai = offset*(2*thid+1)-1;
            unsigned int bi = offset*(2*thid+2)-1;
            if (not slow) {
                ai += CONFLICT_FREE_OFFSET(ai);
                bi += CONFLICT_FREE_OFFSET(bi);
            }
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
        unsigned int * g_cellBegin,
        unsigned int * g_axis,
        unsigned int * g_cellIndices,
        float * g_cuts,
        unsigned int * g_offsetLessEquals,
        unsigned int * g_offsetGreater,
        float * g_particlesX,
        float * g_particlesY,
        float * g_particlesZ,
        float * g_odata,
        unsigned int * g_permutations,
        bool p) {

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
    const unsigned int cellBegin = g_cellBegin[blockIdx.x];
    const unsigned int axis = g_axis[blockIdx.x];

    unsigned int n = blockSize * 2;
    //unsigned int i = begin + 2 * tid;
    //unsigned int j = begin + 2 * tid + 1;

    // slow version but works with all input sizes
    unsigned int l_i = tid;
    unsigned int l_j = tid + n / 2;

    int bankOffsetI = CONFLICT_FREE_OFFSET(l_i);
    int bankOffsetJ = CONFLICT_FREE_OFFSET(l_j);

    unsigned int g_i = begin + l_i;
    unsigned int g_j = begin + l_j;

    l_i += bankOffsetI;
    l_j += bankOffsetJ;

    bool slow = true; //end - begin < n;
    if (slow) {
        l_i = 2 * tid;
        l_j = 2 * tid + 1;
        g_i = begin + l_i;
        g_j = begin + l_j;
    }

    bool f1, f2;
    if (not slow or g_i < end) {
        if (axis == 0) {
            f1 = g_particlesX[g_i] <= cut;
            f2 = not f1;
        }
        else if (axis == 1) {
            f1 = g_particlesY[g_i] <= cut;
            f2 = not f1;
        }
        else {
            f1 = g_particlesZ[g_i] <= cut;
            f2 = not f1;
        }
        // potential to avoid bank conflicts here
        s_lessEquals[l_i] = f1;
        s_greater[l_i] = f2;
    }
    else {
        f1 = false;
        f2 = false;
        s_lessEquals[l_i] = 0;
        s_greater[l_i] = 0;
    }

    bool f3, f4;
    if (not slow or g_j < end) {
        if (axis == 0) {
            f3 = g_particlesX[g_j] <= cut;
            f4 = not f3;
        }
        else if (axis == 1) {
            f3 = g_particlesY[g_j] <= cut;
            f4 = not f3;
        }
        else {
            f3 = g_particlesZ[g_j] <= cut;
            f4 = not f3;
        }
        // potential to avoid bank conflicts here
        s_lessEquals[l_j] = f3;
        s_greater[l_j] = f4;
    }
    else {
        f3 = false;
        f4 = false;
        s_lessEquals[l_j] = 0;
        s_greater[l_j] = 0;
    }

    __syncthreads();

    scan(s_lessEquals, tid, n, slow);
    scan(s_greater, tid, n, slow);

    __syncthreads();

    // Avoid another kernel
    if (tid == blockSize - 1) {
        // result shared among kernel
        // atomicAdd returns old
        // exclusive scan does not include the last element
        s_offsetLessEquals = atomicAdd(&g_offsetLessEquals[cellIndex], s_lessEquals[blockSize * 2 - 1] + f3);
        s_offsetGreater = atomicAdd(&g_offsetGreater[cellIndex], s_greater[blockSize * 2 - 1] + f4);

        if (p) {
            printf("%d %d %d %d %d %d %d %f\n",
                   blockIdx.x,
                   blockSize,
                   cellIndex,
                   nLeft,
                   begin,
                   end,
                   axis,
                   cut);

        }
    }

    __syncthreads();

    // avoiding warp divergence
    unsigned int indexA = cellBegin + (s_lessEquals[l_i] + s_offsetLessEquals) * f1 +
                          (s_greater[l_i] + s_offsetGreater + nLeft) * f2;

    unsigned int indexB = cellBegin + (s_lessEquals[l_j] + s_offsetLessEquals) * f3 +
                          (s_greater[l_j] + s_offsetGreater + nLeft) * f4;

    if (not slow or g_i < end) {
        if (axis == 0) {
            g_odata[indexA] = g_particlesX[g_i];
        }
        else if (axis == 1) {
            g_odata[indexA] = g_particlesY[g_i];
        }
        else {
            g_odata[indexA] = g_particlesZ[g_i];
        }
        //g_odata[i] = indexA;
        g_permutations[g_i] = indexA;
    }

    if (not slow or g_j < end) {
        if (axis == 0) {
            g_odata[indexB] = g_particlesX[g_j];
        }
        else if (axis == 1) {
            g_odata[indexB] = g_particlesY[g_j];
        }
        else {
            g_odata[indexB] = g_particlesZ[g_j];
        }
        //g_odata[j] = indexB;
        g_permutations[g_j] = indexB;
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
    const unsigned int axis = (g_axis[blockIdx.x] + rollAxis) % 3;

    unsigned int i = begin + 2 * tid;
    unsigned int j = begin + 2 * tid + 1;
    //unsigned int gridSize = blockSize*2*gridDim.x;

    if (i < end) {
        if (axis == 0) {
            g_odata[g_permutations[i]] = g_particlesX[i];
        }
        else if (axis == 1) {
            g_odata[g_permutations[i]] = g_particlesY[i];
        }
        else {
            g_odata[g_permutations[i]] = g_particlesZ[i];
        }

        //g_odata[i] = g_permutations[i];
    }

    if (j < end) {
        if (axis == 0) {
            g_odata[g_permutations[j]] = g_particlesX[j];
        }
        else if (axis == 1) {
            g_odata[g_permutations[j]] = g_particlesY[j];
        }
        else {
            g_odata[g_permutations[j]] = g_particlesZ[j];
        }
        //g_odata[j] = g_permutations[j];
    }
}

int ServicePartitionGPU::Service(PST pst,void *vin,int nIn,void *vout, int nOut) {

    auto lcl = pst->lcl;
    auto in  = static_cast<input *>(vin);
    auto out = static_cast<output *>(vout);
    const int nCells = nIn / sizeof(input);

    //int bytes = nCounts * sizeof (uint);
    // https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/

    CUDA_CHECK(cudaMemset, (lcl->d_offsetLeq, 0, sizeof(unsigned int) * nCells));
    CUDA_CHECK(cudaMemset, (lcl->d_offsetG, 0, sizeof(unsigned int) * nCells));

    int nBlocks = 0;
    int blockPtr = 0;
    for (int cellPtrOffset = 0; cellPtrOffset < nCells; ++cellPtrOffset) {
        auto cell = static_cast<Cell>(*(in + cellPtrOffset));
        unsigned int beginInd = pst->lcl->cellToRangeMap(cell.id, 0);
        unsigned int endInd =  pst->lcl->cellToRangeMap(cell.id, 1);
        unsigned int n = endInd - beginInd;

        unsigned int nBlocksPerCell = (int) ceil((float) n / (N_THREADS * 2));

        int begin = beginInd;
        for (int i = 0; i < nBlocksPerCell; ++i) {
            lcl->h_nLefts[blockPtr] = lcl->h_countsLeft(cellPtrOffset);
            //printf("%d\n", lcl->h_nLefts[blockPtr]);
            lcl->h_cellIndices[blockPtr] = cellPtrOffset;
            lcl->h_axis[blockPtr] = cell.cutAxis;
            lcl->h_cuts[blockPtr] = cell.getCut();
            lcl->h_begins[blockPtr] = begin;
            begin += N_THREADS * 2;
            lcl->h_ends[blockPtr] = min(begin, endInd);
            lcl->h_cellBegin[blockPtr] = beginInd;

            /*if (pst->idSelf == 0) {
                printf("ind %i axis %i cut %f b %i e %i nl %i\n",
                       lcl->h_cellIndices[blockPtr],
                       lcl->h_axis[blockPtr],
                       lcl->h_cuts[blockPtr],
                       lcl->h_begins[blockPtr],
                       lcl->h_ends[blockPtr],
                       lcl->h_nLefts[blockPtr]);
            }*/

            blockPtr++;
        }
        nBlocks += nBlocksPerCell;

        out[cellPtrOffset] = 0;
    }

    //printf("nBlocks %i blockptr %i\n", nBlocks, blockPtr);

    // use structs instead
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

    CUDA_CHECK(cudaMemcpyAsync,(
            lcl->d_cellBegin,
            lcl->h_cellBegin,
            sizeof (float) * nBlocks,
            cudaMemcpyHostToDevice,
            lcl->streams(0)));

    CUDA_CHECK(cudaMemset, (lcl->d_offsetLeq, 0, sizeof(unsigned int) * nCells));
    CUDA_CHECK(cudaMemset, (lcl->d_offsetG, 0, sizeof(unsigned int) * nCells));

    bool p = pst->idSelf == 0;
    partition<N_THREADS><<<
        nBlocks,
        N_THREADS,
        N_THREADS * sizeof(unsigned int) * 4 + sizeof(unsigned int) * 2,
        lcl->streams(0)
    >>>(
            lcl->d_begins,
            lcl->d_ends,
            lcl->d_nLefts,
            lcl->d_cellBegin,
            lcl->d_axis,
            lcl->d_cellIndices,
            lcl->d_cuts,
            lcl->d_offsetLeq,
            lcl->d_offsetG,
            lcl->d_particlesX,
            lcl->d_particlesY,
            lcl->d_particlesZ,
            lcl->d_particlesT,
            lcl->d_permutations,
            p
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
            d_to = lcl->d_particlesZ + beginInd;
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
            d_to = lcl->d_particlesX + beginInd;
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

    int offsetLeq[nCells];
    int offsetG[nCells];

    cudaMemcpyAsync(
            offsetLeq,
            lcl->d_offsetLeq,
            sizeof (unsigned int) * nCells,
            cudaMemcpyDeviceToHost,
            pst->lcl->streams(0)
    );
    cudaMemcpyAsync(
            offsetG,
            lcl->d_offsetG,
            sizeof (unsigned int) * nCells,
            cudaMemcpyDeviceToHost,
            pst->lcl->streams(0)
    );

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

    std::vector<int> should;
    std::vector<int> is;

    if (pst->idSelf == 0) {
        for (int i = 0; i < x.rows(); i++) {
            printf("%i: x %f y %f z %f  \n", i, x(i), y(i), z(i));
        }
        printf("\n---------------\n\n");
    }

    if (pst->idSelf == 0) {
        for (int i = 0; i < nCells; ++i) {
            auto cell = static_cast<Cell>(*(in + i));
            printf("%i: %i %i\n", i, offsetLeq[i], offsetG[i]);
            printf("%i: %i %i\n", i, lcl->cellToRangeMap(cell.id, 0), lcl->cellToRangeMap(cell.id, 1));
        };
        printf("\n---------------\n\n");
    }

    return 0;
}

int ServicePartitionGPU::Combine(void *vout,void *vout2,int nIn,int nOut1,int nOut2) {
    return 0;
}
