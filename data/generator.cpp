#include <iostream>
#include <fstream>
#include <blitz/array.h>   

float r01() {
    return (float)(rand())/(float)(RAND_MAX);
}

int main(int argc, char** argv) {
    int rank, np;

    std::cout << "Process " << rank << " processing " << N / 1000 << "K particles." << std::endl;

    // Init positions
    blitz::Array<float, 2> p(N, DIMENSIONS + 1);
    p = 0;
    
    srand(rank);
    for (int i = 0; i < p.rows(); i++) {
        for (int d = 0; d < DIMENSIONS; d++) {
            p(i,d) = (r01()-0.5)*(r01()-0.5);
        }
        p(i,3) = 0.;
    }

    // call orb()
    Orb orb(rank, np);

    std::cout << "Process " << rank << " building tree..." << std::endl;

    auto start = high_resolution_clock::now();
    orb.build(p);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "Process " << rank << " duration: " << duration.count() / 1000 << " ms" << std::endl;

    printf("Done.\n");       

    
    MPI_Finalize();     

    std::fstream file( "../data/splitted" + std::to_string(rank) + ".dat", std::fstream::out);
   
    for (int i = 0; i < N; i += 16){
        file << p(i,0) << "\t" << p(i,1) << "\t" << p(i,2) << "\t" << p(i,3) << std::endl;
    }

    file.close();                           

    return 0;
}
