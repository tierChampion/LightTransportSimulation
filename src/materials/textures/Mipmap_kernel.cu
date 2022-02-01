#ifndef __mipmap_kernel_cu__
#define __mipmap_kernel_cu__

#include "Mipmap.cuh"

namespace lts {

	template <typename T>
	__global__ void pyramidInitKernel(BlockedArray<T>* pyramid, int level) {

		int s = blockIdx.x * blockDim.x + threadIdx.x; // level
		int t = blockIdx.y * blockDim.y + threadIdx.y; // to determine

		(*pyramid[level])(s, t) = 0.25f *
			(texel(2 * s, 2 * t, level - 1, pyramid) + texel(2 * s + 1, 2 * t, level - 1, pyramid) +
				texel(2 * s, 2 * t + 1, level - 1, pyramid) + texel(2 * s + 1, 2 * t + 1, level - 1, pyramid));
	}

}

#endif