#ifndef GPU_LOAD_BALANCE_BASESERVICE_H
#define GPU_LOAD_BALANCE_BASESERVICE_H

#include "../comm/messaging.h"
#include "../orb.h"

#include <vector>
#include <tuple>

class ServiceManager;
class Messaging;

class BaseService {
public:
    const int serviceID = -1;
    std::shared_ptr<ServiceManager> manager;

    virtual void run(const void * inputBuffer,
                     const int nInputElements,
                     void * outputBuffer,
                     int nOutputElements) = 0;

    // In and output data is limited to arrays only
    virtual int getNInputBytes(int inputBufferLength) const = 0;
    virtual int getNOutputBytes(int inputBufferLength) const = 0;

    virtual std::tuple<int, int> getNBytes(int bufferLength) const = 0;

    void setManager(std::shared_ptr<ServiceManager> m) {
        manager = m;
    }
private:

};

#endif //GPU_LOAD_BALANCE_BASESERVICE_H