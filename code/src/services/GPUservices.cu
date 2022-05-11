#include "../orb.h"

__global__ void dCountLeft(
        int nParticles,
        float * particles,
        int nCuts,
        float * cuts,
        int * counts
        ) {
    return;
}



void hCountLeft(Orb &orb, Cell * c, int * results, int n) {

    /*unsigned int nThreads = 256;
    int nBlocks = (orb.particles.rows() + nThreads - 1) / nThreads;

    // Device memory
    int size_particles = orb.particles.size();
    float* d_particles;
    cudaMalloc(&d_particles, size_particles);

    // Copy to device
    cudaMemcpy(d_particles, orb.particles.data(), size_particles, cudaMemcpyHostToDevice);

*/


}