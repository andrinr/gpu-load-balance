#ifndef ORB_H // include guard
#define ORB_H

#include <tuple>

class Orb {
public:
    float* particles;
    
    Orb(float* particles);
    void build();

private:
    void reshuffleArray(int axis, int begin, int end, float split);
    int countLeft(int axis, int start, int end, float cut);
    std::tuple<float, int> findCut(int axis, int begin, int end, float left, float right);
    void operative();
    void worker();
};

#endif //ORB_H