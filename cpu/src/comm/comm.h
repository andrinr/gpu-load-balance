#ifndef COMM_H // include guard
#define COMM_H

template <class T>
class Comm {
public:
    Comm();

    T static dispatchService(
            T (*func)(T),
            T* inData,
            int nInData,
            T* outData,
            int nOutData,
            int source);

    static void signalDataSize(int size);
    static void signalServiceId(int flag);

    void destroy();

};

#endif //COMM_H