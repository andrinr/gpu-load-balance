#ifndef CODE_COUNTSERVICE_H
#define CODE_COUNTSERVICE_H

struct ServiceContext {
    Orb& orb,

};

class BaseService {
public:
    static const int serviceID = -1;

    BaseService();

    virtual void run(void * rawInputData, void * rawOutputData) = 0;
};

#endif //CODE_BASESERVICE_H