#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
namespace Thrust {

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
    // TODO use `thrust::exclusive_scan`
    // example: for device_vectors dv_in and dv_out:
    // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
	thrust::host_vector<int> hst_in(idata, idata+n);
	thrust::device_vector<int> dev_in(hst_in);
	thrust::device_vector<int> dev_out(n);

	cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

	thrust::exclusive_scan(dev_in.begin(), dev_in.end(), dev_out.begin());

	cudaEventRecord(stop);
    cudaEventSynchronize(stop);
	float elapsedTime; 
    cudaEventElapsedTime(&elapsedTime , start, stop);
	printf("time is %f ms on the GPU\n", elapsedTime);
	
	thrust::copy(dev_out.begin(), dev_out.end(), odata);
}

}
}
