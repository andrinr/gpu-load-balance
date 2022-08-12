# Orthogonal Recursive Bisection on the GPU for Accelerated Load Balancing in Large N-Body Simulations

Bachelor Thesis of Andrin Rehmann @UZH 2022.


The N-Body technique has been used for decades to simulate the Universe to compare theory with observations. It uses \textit{particles} to represent objects of a certain mass and computes the interacting forces by applying the gravity equation. As the forces operate over an infinite distance it is necessary to consider all pairwise interactions, making a naive implementation $\mathcal{O}(N^2)$ where $N$ is the number of particles in the system. This does not scale with large $N$'s, in fact the problem becomes computationally infeasible at some point.

A common solution is to partition the space, in which the particles are contained, into a set of subspaces using the Orthogonal Recursive Bisection (ORB) algorithm. These subspaces, or cells, are then stored in a space partitioning tree data structure (SPTDS). The data structure is then leveraged in combination with the Fast Multipole Method (FMM) to speed up the particle simulation to $O(n)$. In order to balance the load of work and memory across nodes and processors in a computing system, a load balancing is necessary. The same tree, which is used for FMM can be leveraged to generate groupings of the particles which are then distributed among processing units. Building the SPTDS with ORB uses a significant percentage of the overall simulation time, \cite{Stadel2001} where as of now this was done on the CPU, not leveraging GPU acceleration.
 
In this thesis I established an upper limit for the speedup of a GPU ORB implementation over its fully parallelized CPU only counterpart. In order to do so, I proposed a runtime estimate for both versions. As hardware specific details and some knowledge about compilers and Assembly instructions are crucial for a reliable estimate, I explored the theoretical foundations of memory bandwidth and processing power. 

Based on encouraging figures from the runtime estimates I proceeded with the actual implementation. Using the machine dependent layer (MDL) from PKDGRAV \cite{Stadel2001}, which is used to distribute workload among cores and processors, I implemented a fully parallelized CPU version of ORB. In preparation of advancing parts of the code to a GPU accelerated implementation, I have summarized the most important and relevant concepts of CUDA, a popular graphics programming language developed by NVIDIA. Compared to traditional C++ code, many hardware related limitations have to be considered and a fundamentally different paradigms is required.

With the gained knowledge I implemented the most performance critical part of ORB using CUDA \cite{CUDAGuide}. The identified part is essentially a map and reduce on the particles coordinates. I iteratively increased the performance of the kernels and describe each version in detail.

Moreover I propose a possible method to accelerate the partitioning part of ORB with specific details. The proposed implementation could possibly further decrease the runtime of ORB when implemented.

Finally, I have performed a runtime analysis on Piz Daint \cite{piz_daint}, a powerful super computer in Switzerland. A significant speedup of the GPU accelerated ORB over its CPU counterpart was found. 


Read the entire [/documentation](https://github.com/andrinr/gpu-load-balance/tree/main/documentation).

## Get started

1. Clone https://bitbucket.org/dpotter/pkdgrav3/
2. Clone this repo and ``cd`` into its root directory
3. Link mdl2 and blitz from the PKDGRAV repo using ``ln -s /path/to/mdl2`` ``ln -s /path/to/blitz``

### Compile
1. ``mkdir release && cd release``
3. ``cmake ..``
4. `` cmake --build .``

### Run
``./orbit`` <x> <y> <o>
where x corresponds to the number of particles 2^x and y to the number of leaf cells in the final tree datastrucutre 2^y. o defines a gpu optimization level where 0 or empty is a cpu only version, 1 corresponds to gpu accelerated version where the bisection method is implented using kernels. Finally 2 further adds a GPU accelerated partitioning method, however this is is still in experimental stages and some bugs are still present.


### Compile for debugging
1. ``mkdir debug && cd debug``
2. ``cmake -DCMAKE_BUILD_TYPE=Debug --DBZ_DEBUG ..``
3. `` cmake --build .``


### Debug
``gdb ./orbit``
or
``cuda-gdb ./orbit``


## Automized performance analysis

For slurm process manager run ``analysis/analysis.sh`` for a test with variable particle counts and ``analysis/analysis2.sh`` for a test with variable domain counts. 
