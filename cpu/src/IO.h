#ifndef IO_H
#define IO_H
#include <blitz/array.h>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include "constants.h"

class IO {
public:
    static blitz::Array<float, 2> generateData(int N, int seed) {

        // Init positions
        blitz::Array<float, 2> p(N, DIMENSIONS + 1);
        p = 0;

        srand(seed);
        for (int i = 0; i < p.rows(); i++) {
            for (int d = 0; d < DIMENSIONS; d++) {
                p(i,d) = (r01()-0.5)*(r01()-0.5);
            }
            p(i,3) = 0.;
        }

        return p;
    }

    static blitz::Array<float, 2> loadData(int N) {
        blitz::Array<float, 2> p(N, DIMENSIONS + 1);

        return p;
    }

    static void logMeasurements(std::string output) {
        std::filesystem::path cwd = std::filesystem::current_path() / (output);
        std::ofstream file(cwd.string(), std::fstream::app);

        // todo
        //file << g_duration << "," << count << "," << np << std::endl;

        file.close();
    }

    static void writeData(blitz::Array<float, 2> p, int rank) {
        std::fstream file( "out/splitted" + std::to_string(rank) + ".dat", std::fstream::out);

        for (int i = 0; i < p.rows(); i += 64){
            file << p(i,0) << "\t" << p(i,1) << "\t" << p(i,2) << "\t" << p(i,3) << std::endl;
        }

        file.close();
    }

private:
    static float r01() {
        return (float)(rand())/(float)(RAND_MAX);
    }
};


#endif //IO_H
