//
// Created by andrin on 18/05/22.
//

#ifndef GPU_LOAD_BALANCE_SERVICEMANAGER_H
#define GPU_LOAD_BALANCE_SERVICEMANAGER_H

#include "baseService.h"
#include <map>
#include "../comm/messaging.h"

class Messaging;

class ServiceManager {
public:
    std::map<int, BaseService * > m;

    ServiceManager(Orb * o, Messaging * m) {
        orb = o;
        messaging = m;
    }

    void addService(BaseService * service) {
        m[service->serviceID] = service;
        service->setManager(this);
    }

    Orb * orb;
    Messaging * messaging;
};


#endif //GPU_LOAD_BALANCE_SERVICEMANAGER_H
