#pragma once

#include <cstdio>
#include <cstring>
#include <cmath>

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
#define blockSize 256

/**
 * Check for CUDA errors; print and exit if there was a problem.
 */
void checkCUDAErrorFn(const char *msg, const char *file = NULL, int line = -1);

void cpyHostToDevice(const int *hst, int *dev, size_t array_size);
void cpyDeviceToHost(int *hst, const int *dev, size_t array_size);
void cpyDeviceToDevice(const int *src, int *dest, size_t array_size);

inline int power(int x, int y) {
	int result = 1;
	while(y > 0) {
		result *= x;
		--y;
	}
	return result;
}

inline int ilog2(int x) {
    int lg = 0;
    while (x >>= 1) {
        ++lg;
    }
    return lg;
}

inline int ilog2ceil(int x) {
    return ilog2(x - 1) + 1;
}


namespace StreamCompaction {
namespace Common {
    __global__ void kernMapToBoolean(int n, int *bools, const int *idata);

    __global__ void kernScatter(int n, int *odata,
            const int *idata, const int *bools, const int *indices);
}
}
