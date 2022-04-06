#include <blitz/array.h>   
#include <chrono>
using namespace std::chrono;

static const int N = 1 << 30;

int main(int argc, char** argv) {

    // Init positions
    blitz::Array<float, 1> p(N);
    p = 0;

    for (int i = 0; i < N; i++) {
        p(i) = (float)rand()/(float)(RAND_MAX) - 0.5;
    }

    auto start = high_resolution_clock::now();

    float c = 0;
    int n_left = 0;
    for (int i = 0; i < N; i++) {
        c += p(i) < c;
    }

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "with memory access: " << duration.count() << std::endl;

    start = high_resolution_clock::now();

    c = 0;
    n_left = 0;
    for (int i = 0; i < N; i++) {
        c += 1 < c;
    }

    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    std::cout << "without memory access: " << duration.count() << std::endl;

}