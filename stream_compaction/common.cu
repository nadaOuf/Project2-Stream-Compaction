#include "common.h"

void checkCUDAErrorFn(const char *msg, const char *file, int line) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
}

void cpyHostToDevice(const int *hst, int *dev, size_t array_size) {
	cudaMemcpy(	/*destination*/ dev,
				/*source*/ hst,
				/*size in bytes to copy*/ array_size,
				/*cudaMemcpy type*/ cudaMemcpyHostToDevice);

	checkCUDAError("Error copying memory from host to device");	
}

/**
*/
void cpyDeviceToHost(int *hst,const int *dev, size_t array_size) {
	cudaMemcpy(	/*destination*/ hst,
				/*source*/ dev,
				/*size in bytes to copy*/ array_size,
				/*cudaMemcpy type*/ cudaMemcpyDeviceToHost);

	checkCUDAError("Error copying memory from device to host");
}

/**
*/
void cpyDeviceToDevice(const int *src, int *dest, size_t array_size) {
	cudaMemcpy(	/*destination*/ dest,
				/*source*/ src,
				/*size in bytes to copy*/ array_size,
				/*cudaMemcpy type*/ cudaMemcpyDeviceToDevice);

	checkCUDAError("Error copying memory from device to device");
}


namespace StreamCompaction {
namespace Common {

/**
 * Maps an array to an array of 0s and 1s for stream compaction. Elements
 * which map to 0 will be removed, and elements which map to 1 will be kept.
 */
__global__ void kernMapToBoolean(int n, int *bools, const int *idata) {
    int index = (blockIdx.x*blockDim.x) + threadIdx.x;
	
	if(index < n) {
		bools[index] = (idata[index] != 0);	
	}
}

/**
 * Performs scatter on an array. That is, for each element in idata,
 * if bools[idx] == 1, it copies idata[idx] to odata[indices[idx]].
 */
__global__ void kernScatter(int n, int *odata,
        const int *idata, const int *bools, const int *indices) {
    int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	if(index < n && bools[index]) {
		odata[indices[index]] = idata[index];
	}
}

}
}
