//
// Created by andrin on 19/05/22.
//

#include "statusService.h"

void StatusService::run(void *rawInputData, void *rawOutputData) {
    rawOutputData = rawOutputData;
}

void StatusService::getNInputBytes(void *inputPtr) const {
    return sizeof(int);
}

void StatusService::getNOutputBytes(void *outputPtr) const {
    return sizeof(int);
}