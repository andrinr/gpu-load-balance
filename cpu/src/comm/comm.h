#ifndef COMM_H // include guard
#define COMM_H


class Comm {
public:
    void Comm();

    virtual void static dispatchWork(int& n, Cell* cells);

    virtual void static concludeWork(int&n, int* count);

    virtual void destroy();

};

#endif //COMM_H