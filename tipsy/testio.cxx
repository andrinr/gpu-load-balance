#include "tipsy.h"

int main() {
    TipsyIO io;

    io.open("b0-final.std");
    std::cout << io.count() << std::endl;

    if (io.fail()) {
        std::cerr << "Unable to open file" << std::endl;
        abort();
    }


    blitz::Array<float,2> r(io.count(),3);
    io.load(r);
    std::cout << r << std::endl;


}
