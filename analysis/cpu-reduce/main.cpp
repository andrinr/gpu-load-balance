//
// Created by andrin on 5/30/22.
//

#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

int sum(float * particles, int n, int k) {

    int nLeft = 0;
    float cut = 0.5;

    float * startPtr = particles + k;
    float * endPtr = startPtr + n;

    // We use pointers to iterate over the particles
    for(auto p= startPtr; p<endPtr; ++p) nLeft += *p < cut;

    return nLeft;
}

int main(int argc, char** argv) {
    
    int n = 1 << strtol(argv[1], nullptr, 0);;

    float * particles = (float *) malloc(n * sizeof(float));

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

    printf("%u %u \n", nThreads,  std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

    free(particles);

    return 0;
}
