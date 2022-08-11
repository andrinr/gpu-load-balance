#include <blitz/array.h>
#include <vector>


#define FULL_MASK 0xffffffff
template <unsigned int blockSize>
extern __device__ void warpReduce2(volatile unsigned int *s_data, unsigned int tid) {
    if (blockSize >= 64) s_data[tid] += s_data[tid + 32];
    __syncwarp();
    if (tid < 32) {
        unsigned int val = s_data[tid];
        for (int offset = warpSize/2; offset > 0; offset /= 2)
            val += __shfl_down_sync(FULL_MASK, val, offset);
        if (tid == 0) {
            s_data[0] = val;
        }
    }
}

template <unsigned int blockSize>
extern __global__ void reduce(
        float * g_idata,
        unsigned int * g_begins,
        unsigned int * g_ends,
        float * g_cuts,
        unsigned int * g_odata) {

    __shared__ unsigned int s_data[blockSize];

    unsigned int tid = threadIdx.x;
    const unsigned int begin = g_begins[blockIdx.x];
    const unsigned int end = g_ends[blockIdx.x];
    const float cut = g_cuts[blockIdx.x];

    unsigned int i = begin + tid;
    s_data[tid] = 0;

    // unaligned coalesced g memory access
    while (i < end) {
        s_data[tid] += (g_idata[i] <= cut);
        i += blockSize;
    }
    __syncthreads();

    if (blockSize >= 512) {
        if (tid < 256) {
            s_data[tid] += s_data[tid + 256];
        }
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) {
            s_data[tid] += s_data[tid + 128];
        } __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) {
            s_data[tid] += s_data[tid + 64];
        } __syncthreads();
    }
    if (tid < 32) {
        warpReduce2<blockSize>(s_data, tid);
    }
    if (tid == 0) {
        g_odata[blockIdx.x] = s_data[0];
    }
}

template <unsigned int blockSize>
extern __global__ void reduce2(
        float * g_idata,
        unsigned int * g_begins,
        unsigned int * g_ends,
        float * g_cuts,
        unsigned int * g_odata) {

    __shared__ unsigned int s_data[blockSize];

    unsigned int tid = threadIdx.x;
    const unsigned int begin = g_begins[blockIdx.x];
    const unsigned int end = g_ends[blockIdx.x];
    const float cut = g_cuts[blockIdx.x];

    unsigned int i = begin + tid;
    s_data[tid] = 0;

    unsigned int val = 0;
    // unaligned coalesced g memory access
    while (i < end) {
        val += (g_idata[i] <= cut);
        i += blockSize;
    }
    __syncwarp();

    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(FULL_MASK, val, offset);

    s_data[tid] = val;
    __syncthreads();

    if (tid > 32 ) {
        return;
    }

    val = 0;
    if (tid < 32 && tid * 32 < blockSize) {
        val += s_data[tid * 32];
    }
    __syncwarp();

    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(FULL_MASK, val, offset);

    if (tid == 0) {
        g_odata[blockIdx.x] = val;
    }
}


template <unsigned int blockSize>
extern __global__ void reduce3(
        float * g_idata,
        unsigned int * g_begins,
        unsigned int * g_ends,
        float * g_cuts,
        unsigned int * g_odata) {

    __shared__ unsigned int s_data[32];

    unsigned int tid = threadIdx.x;
    const unsigned int begin = g_begins[blockIdx.x];
    const unsigned int end = g_ends[blockIdx.x];
    const float cut = g_cuts[blockIdx.x];
    const unsigned int lane = tid & (32 - 1);
    const unsigned int warpId = tid >> 5;
    unsigned int i = begin + tid;
    s_data[tid] = 0;

    //printf("%i %i %i %i %i %i\n", tid, lane, warpId, begin, end, blockSize);

    unsigned int val = 0;
    // unaligned coalesced g memory access
    while (i < end) {
        val += (g_idata[i] <= cut);
        i += blockSize;
    }
    __syncwarp();

    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(FULL_MASK, val, offset);

    if (lane == 0) {
        s_data[warpId] = val;
    }
    __syncthreads();

    // All warps but first one are not needed anymore
    if (warpId == 0) {
        val = 0;
        if (tid < 32 && tid * 32 < blockSize) {
            val += s_data[tid];
        }

        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(FULL_MASK, val, offset);

        if (tid == 0) {
            g_odata[blockIdx.x] = val;
        }
    }
}

int main(int argc, char** argv) {

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int nThreads = 256;
    int elementsPerThread = 16;

    unsigned int n = 1 << 28;
    unsigned int d = 1 << 10;
    unsigned int maxBlocks = n / (elementsPerThread * nThreads) + d;

    float * pos = (float*)calloc(n, sizeof(float));
    for (unsigned int i = 0; i < n; i++) {
        pos[i] = (float)(rand())/(float)(RAND_MAX);
    }

    unsigned int * h_beginIds = (unsigned int*)calloc(maxBlocks, sizeof(unsigned int));
    unsigned int * h_endIds = (unsigned int*)calloc(maxBlocks, sizeof(unsigned int));
    float * h_cuts = (float*)calloc(maxBlocks, sizeof(float));
    unsigned int * cellIds = (unsigned int*)calloc(maxBlocks, sizeof(unsigned int));

    int nBlocks = 0;
    int blockPtr = 0;
    for (unsigned int i = 0; i < d; i++) {

        int begin = n / d * i;
        int endInd = n / d * (i + 1);

        unsigned int nBlocksPerCell =
                max((int) floor((float) (endInd - begin) / (nThreads * elementsPerThread)),1);

        //printf("%d %d %d %d\n", i, begin, endInd, nBlocksPerCell);

        for (int j = 0; j < nBlocksPerCell; ++j) {
            cellIds[blockPtr] = i;
            h_beginIds[blockPtr] = begin;
            begin += nThreads * elementsPerThread;
            if (j == nBlocksPerCell - 1) {
                h_endIds[blockPtr] = endInd;
            } else {
                h_endIds[blockPtr] = begin;
            }
            //printf("wat %i %i %i\n", h_beginIds[blockPtr], h_endIds[blockPtr], cellIds[blockPtr]);
            h_cuts[blockPtr] = 0.5;
            blockPtr++;
        }

        nBlocks += nBlocksPerCell;
    }

    printf("%d\n", nBlocks);
    unsigned int * testSums = (unsigned int*)calloc(d, sizeof(unsigned int));
    for (unsigned int i = 0; i < nBlocks; i++) {

        for (unsigned int j = h_beginIds[i]; j < h_endIds[i]; j++) {
            testSums[cellIds[i]] += (pos[j] <= h_cuts[i]);
        }
    }

    printf("computed test sums\n");

    float * d_particles;
    unsigned int * d_sums;
    unsigned int * h_sums = (unsigned int*)calloc(n, sizeof(unsigned int));
    unsigned int * d_begins;
    unsigned int * d_ends;
    float * d_cuts;
    cudaMalloc(&d_particles, sizeof (float) * n);
    cudaMemcpy(d_particles, pos, sizeof (float ) * n, cudaMemcpyHostToDevice);

    cudaMalloc(&d_sums, sizeof (unsigned int) * nBlocks);

    cudaMalloc(&d_begins, sizeof (unsigned int) * nBlocks);
    cudaMalloc(&d_ends, sizeof (unsigned int) * nBlocks);
    cudaMalloc(&d_cuts, sizeof (float) * nBlocks);
    cudaMemcpy(d_begins, h_beginIds, sizeof (unsigned int) * nBlocks, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ends, h_endIds, sizeof (unsigned int) * nBlocks, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cuts, h_cuts, sizeof (float) * nBlocks, cudaMemcpyHostToDevice);

    printf("copied data to device\n");

    cudaEventRecord(start);
    reduce3<nThreads><<<
            nBlocks,
            nThreads,
            nThreads * sizeof (unsigned int) >>>
            (d_particles,
            d_begins,
            d_ends,
            d_cuts,
            d_sums);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    printf("reduced data\n");
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << milliseconds << "\n";

    cudaMemcpy(h_sums, d_sums, sizeof (int ) * nBlocks, cudaMemcpyDeviceToHost);


    unsigned int * testSums2 = (unsigned int*)calloc(d, sizeof(unsigned int));
    for (int i = 0; i < nBlocks; ++i) {
        testSums2[cellIds[i]] += h_sums[i];
    }

    for (unsigned int i = 0; i < d; i++) {
        if (testSums[i] != testSums2[i]) {
            printf("invalid: cell %d should be %d is %d\n", i, testSums[i], testSums2[i]);
        }
    }

    cudaFree(d_particles);
    cudaFree(d_sums);
    cudaFree(d_begins);
    cudaFree(d_ends);
    cudaFree(d_cuts);

    free(h_sums);
    free(h_beginIds);
    free(h_endIds);
    free(h_cuts);
    free(pos);

    cudaDeviceReset();
}
