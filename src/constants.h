//
// Created by andrin on 6/21/22.
//

#ifndef GPU_LOAD_BALANCE_CONSTANTS_H
#define GPU_LOAD_BALANCE_CONSTANTS_H

static const int MAX_CELLS = 8192;
static const int N_THREADS = 512;
static const int ELEMENTS_PER_THREAD = 32;
static const int N = 1 << 26;
static const int d = 1 << 14;

#endif //GPU_LOAD_BALANCE_CONSTANTS_H
