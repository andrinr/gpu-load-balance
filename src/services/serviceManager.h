//
// Created by andrin on 18/05/22.
//

#ifndef GPU_LOAD_BALANCE_SERVICEMANAGER_H
#define GPU_LOAD_BALANCE_SERVICEMANAGER_H

#include <map>
#include <memory>

#include "baseService.h"
#include "../comm/messaging.h"

class Messaging;
class Orb;

class ServiceManager {
public:
    std::map<int, std::unique_ptr<BaseService>> m;

    ServiceManager() {

    }

    ServiceManager(std::shared_ptr<Orb> o, std::shared_ptr<Messaging> m) {
        orb = o;
        messaging = m;
    }

    void addService(std::unique_ptr<BaseService> service) {
        m[service->serviceID] = std::move(service);
    }

    std::shared_ptr<Orb> orb;
    std::shared_ptr<Messaging> messaging;
};


#endif //GPU_LOAD_BALANCE_SERVICEMANAGER_H
