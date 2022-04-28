#ifndef IO_H
#define IO_H

class IO {
public:
    static blitz::Array<float, 2> generateData(int N) {
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

        return p;
    }

    static blitz::Array<float, 2> loadData(int N) {
        return Null;
    }

    static void logMeasurements(String output) {
        std::filesystem::path cwd = std::filesystem::current_path() / (argv[3]);
        std::ofstream file(cwd.string(), std::fstream::app);

        file << g_duration << "," << count << "," << np << std::endl;

        file.close();
    }

    static void writeData(blitz::Array<float, 2>) {
        std::fstream file( "out/splitted" + std::to_string(rank) + ".dat", std::fstream::out);

        for (int i = 0; i < N; i += 64){
            file << p(i,0) << "\t" << p(i,1) << "\t" << p(i,2) << "\t" << p(i,3) << std::endl;
        }

        file.close();
    }
};


#endif //IO_H
