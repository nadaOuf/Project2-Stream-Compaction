#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
namespace Naive {

int *dev_idata;
int *dev_odata;

__global__ void kernNaive(int n, int step, int *odata, int *idata) {
	
	int index = (blockIdx.x *blockDim.x) + threadIdx.x;
	
	odata[0] = idata[0];
	if(index < n) {
		if(index >= step) {
			odata[index] = idata[index - step] + idata[index];
		} else {
			odata[index] = idata[index];
		}
	}
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *hst_odata, const int *hst_idata) {
	
	//The number of blocks
    dim3 blocksPerGrid((n + blockSize - 1)/blockSize);
	
	//The number of threads per block
	dim3 threadsPerBlock(blockSize);
	
	//Allocate memory
	cudaMalloc((void**)&dev_idata, n*sizeof(int));
	checkCUDAError("Error in malloc of dev_idata");

	cudaMalloc((void**)&dev_odata, n*sizeof(int));
	checkCUDAError("Error in malloc of dev_odata");

	//copy array to device
	cpyHostToDevice(hst_idata, dev_idata, n*sizeof(int));

	int passes = ilog2ceil(n);
	
	cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

	for(int d = 1; d <= passes; ++d) {
		if(d%2) {
			kernNaive<<<blocksPerGrid, threadsPerBlock>>>(n, power(2, d - 1), dev_odata, dev_idata);
		} else {
			kernNaive<<<blocksPerGrid, threadsPerBlock>>>(n, power(2, d - 1), dev_idata, dev_odata);
		}
	}

	cudaEventRecord(stop);
    cudaEventSynchronize(stop);
	float elapsedTime; 
    cudaEventElapsedTime(&elapsedTime , start, stop);
	printf("time is %f ms on the GPU\n", elapsedTime);
	
	cpyDeviceToHost(hst_odata+1, dev_idata, (n-1)*sizeof(int));
	hst_odata[0] = 0;
}

}
}
