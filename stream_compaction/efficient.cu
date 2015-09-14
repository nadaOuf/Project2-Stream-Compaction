#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
namespace Efficient {

__global__ void kernUpSweep(int n, int d, int offset, int *idata) {
	int index = (blockIdx.x *blockDim.x) + threadIdx.x;
	
	if(index < d) {
		int ai = offset*((index*2)+1) - 1;
		int bi = offset*((index*2)+2) - 1;
		if(ai >= n || bi >= n)
			return;
		if(bi == n-1)
			idata[bi] = 0;
		else
			idata[bi] += idata[ai];
	}
}

__global__ void kernDownSweep(int n, int d, int offset, int *idata) {
	int index = (blockIdx.x *blockDim.x) + threadIdx.x;
	
	if(index < d) {
		int ai = offset*((index*2)+1) - 1;
		int bi = offset*((index*2)+2) - 1;
		
		int temp = idata[ai];
		idata[ai] = idata[bi];
		idata[bi] += temp;
	}

}


/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *hst_odata, const int *hst_idata) {
 	int *dev_idata;

	//The number of blocks
    dim3 blocksPerGrid((n + blockSize -1)/blockSize);
	
	//The number of threads per block
	dim3 threadsPerBlock(blockSize);
	
	//Allocate memory
	cudaMalloc((void**)&dev_idata, n*sizeof(int));
	checkCUDAError("Error in malloc of dev_idata");
	
	//copy array to device
	cpyHostToDevice(hst_idata, dev_idata, n*sizeof(int));

	int passes = ilog2ceil(n);
	
	int offset = 1;
	int d = power(2, passes)>>1;

	cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

	//Up sweep
	for(; d > 0; d >>= 1) {
		kernUpSweep<<<blocksPerGrid, threadsPerBlock>>>(n, d, offset, dev_idata);
		offset *= 2;
	}

	//Down sweep
	for(d = 1; d < n; d *= 2) {
		offset >>= 1;
		kernDownSweep<<<blocksPerGrid, threadsPerBlock>>>(n, d, offset, dev_idata);
	}

	cudaEventRecord(stop);
    cudaEventSynchronize(stop);
	float elapsedTime; 
    cudaEventElapsedTime(&elapsedTime , start, stop);
	printf("time is %f ms on the GPU\n", elapsedTime);

	cpyDeviceToHost(hst_odata, dev_idata, (n*sizeof(int)));

}

/**
 * Performs stream compaction on idata, storing the result into odata.
 * All zeroes are discarded.
 *
 * @param n      The number of elements in idata.
 * @param odata  The array into which to store elements.
 * @param idata  The array of elements to compact.
 * @returns      The number of elements remaining after compaction.
 */
int compact(int n, int *hst_odata, const int *hst_idata) {
    int *dev_idata;
	int *dev_odata;
	int *dev_bools;
	int *dev_indices;
	
	//The number of blocks
    dim3 blocksPerGrid((n + blockSize -1)/blockSize);
	
	//The number of threads per block
	dim3 threadsPerBlock(blockSize);
	
	//Allocate memory
	cudaMalloc((void**)&dev_idata, n*sizeof(int));
	checkCUDAError("Error in malloc of dev_idata");

	cudaMalloc((void**)&dev_odata, n*sizeof(int));
	checkCUDAError("Error in malloc of dev_odata");

	cudaMalloc((void**)&dev_bools, n*sizeof(int));
	checkCUDAError("Error in malloc of dev_bools");

	cudaMalloc((void**)&dev_indices, n*sizeof(int));
	checkCUDAError("Error in malloc of dev_indices");

	//copy array to device
	cpyHostToDevice(hst_idata, dev_idata, n*sizeof(int));
	
	Common::kernMapToBoolean<<<blocksPerGrid, threadsPerBlock>>>(n, dev_bools, dev_idata);
	
	int *hst_bools = new int[n];
	int *hst_indices = new int[n];

	cpyDeviceToHost(hst_bools, dev_bools, (n*sizeof(int)));

	scan(n, hst_indices, hst_bools);

	cpyHostToDevice(hst_indices, dev_indices, n*sizeof(int));	

	Common::kernScatter<<<blocksPerGrid, threadsPerBlock>>>(n, dev_odata, dev_idata, dev_bools, dev_indices);

	cpyDeviceToHost(hst_odata, dev_odata, n*sizeof(int));

	return hst_bools[n-1] == 1 ? hst_indices[n-1]+1 : hst_indices[n-1];
}

}
}
