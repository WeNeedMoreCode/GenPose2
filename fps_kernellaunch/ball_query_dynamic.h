#pragma once

struct BallQueryTilingData {
    int32_t B;
    int32_t N;
    int32_t M;
    int32_t nsample;
    float radius;
    int32_t numCores;
    int32_t queriesPerCore;
};
