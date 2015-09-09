#include <cstdio>
#include "cpu.h"

namespace StreamCompaction {
namespace CPU {

/**
 * CPU scan (prefix sum).
 */
void scan(int n, int *odata, const int *idata) {
    
	odata[0] = 0;
	
	for(int i = 1; i < n; ++i) {
		odata[i] = idata[i-1] + odata[i-1];
	}

}

/**
 * CPU stream compaction without using the scan function.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithoutScan(int n, int *odata, const int *idata) {
    
	int compactN = 0;
	for(int i = 0; i < n; ++i) {
		if(idata[i] != 0) {
			odata[compactN] = idata[i];
			++compactN;
		}
	}

    return compactN;
}

/**
 * CPU stream compaction using scan and scatter, like the parallel version.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithScan(int n, int *odata, const int *idata) {
    int compactN = 0;
	int *intermediate = new int[n];
	int *scanResult = new int[n];

	for(int i = 0; i < n; ++i) {
		if(idata[i] != 0) {
			intermediate[i] = 1;
		} else {
			intermediate[i] = 0;
		}
	}

	scan(n, scanResult, intermediate);

	for(int i = 0; i < n; ++i) {
		if(intermediate[i] == 1) {
			odata[scanResult[i]] = idata[i];
		}
	}

	compactN = scanResult[n-1];
	if(intermediate[n-1] == 1) {
		++compactN;
	}

    return compactN;
}

}
}
