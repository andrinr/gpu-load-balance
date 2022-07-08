//
// Created by andrin on 6/21/22.
//

#ifndef GPU_LOAD_BALANCE_CONSTANTS_H
#define GPU_LOAD_BALANCE_CONSTANTS_H

static const int MAX_CELLS = 8192;
static const int N_THREADS = 512;
static const int ELEMENTS_PER_THREAD = 32;
static const int N = 1 << 25;
static const int d = 1 << 4;
static const int N_STREAMS = 32;

enum GPU_ACCELERATION {
    NONE,
    COUNT,
    COUNT_PARTITION
};

inline void CUDA_Abort(cudaError_t rc, const char *fname, const char *file, int line) {
    fprintf(stderr,"%s error %d in %s(%d)\n%s\n", fname, rc, file, line, cudaGetErrorString(rc));
    exit(1);
}
#define CUDA_CHECK(f,a) {cudaError_t rc = (f)a; if (rc!=cudaSuccess) CUDA_Abort(rc,#f,__FILE__,__LINE__);}

#endif //GPU_LOAD_BALANCE_CONSTANTS_H
