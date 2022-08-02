//
// Created by andrin on 8/1/22.
//


#include "../src/services/partitionGPU.h"
#include "catch2.h"

TEST_CASE("function1", "function1")
{
    REQUIRE_FALSE(function1(0));
    REQUIRE_FALSE(function1(5));
    REQUIRE(function1(10));
}