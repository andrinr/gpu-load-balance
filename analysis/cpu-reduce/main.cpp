//
// Created by andrin on 5/30/22.
//

#include <blitz/array.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

int sum(blitz::Array<float, 2> particles, int n, int k) {

    int nLeft = 0;
    float cut = 0.5;

    float * startPtr = particles.data() + k;
    float * endPtr = startPtr + n;

    // We use pointers to iterate over the particles
    for(auto p= startPtr; p<endPtr; ++p) nLeft += *p < cut;



    return nLeft;
}

int main(int argc, char** argv) {
    
    int n = 1 << strtol(argv[1], nullptr, 0);;
    int k = 10;
    std::cout << "Performing measurements " << n << "\n";

    // Set row major -> can enable AVX
    blitz::GeneralArrayStorage<2> storage;
    storage.ordering() = 0,1;
    storage.base() = 0, 0;
    storage.ascendingFlag() = true, true;

    blitz::Array<float, 2> particles = blitz::Array<float, 2>(n, k);

    float * startPtr = particles.data();
    float * endPtr = startPtr + n * k;

    //for(auto p= startPtr; p<endPtr; ++p) *p = (float)(rand())/(float)(RAND_MAX);;

    int nThreads = strtol(argv[2], nullptr, 0);;
    std::vector<std::thread> threads;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < nThreads; i++) {
        threads.push_back(std::thread(sum, particles, n / nThreads , n / nThreads * i));
    }

    for (int i = 0; i < nThreads; i++) {
        threads[i].join();
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "cost in microseconds with c style pointer iteration " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "\n";


    return 0;
}