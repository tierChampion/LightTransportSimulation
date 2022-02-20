#ifndef __mipmap_kernel_cu__
#define __mipmap_kernel_cu__

#include "Mipmap.cuh"

namespace lts {

	__global__ void pyramidBaseInit(BlockedArray<Spectrum>* pyramid, int width, int height, Spectrum* image, int levels) {

		int i = blockIdx.x * blockDim.x + threadIdx.x;

		if (i > 0) return;

		pyramid[0] = BlockedArray<Spectrum>(width, height, image);

		for (int l = 1; l < levels; l++) {
			pyramid[l] = BlockedArray<Spectrum>(fmaxf(1, width >> l), fmaxf(1, height >> l));
			pyramid[l].prepareData();
		}
	}

	__global__ void pyramidInitKernel(BlockedArray<Spectrum>* pyramid, int level, ImageWrap mode) {


		int s = blockIdx.x * blockDim.x + threadIdx.x;
		int t = blockIdx.y * blockDim.y + threadIdx.y;

		if (s < 0 || s >= (16 >> level)) return;
		if (t < 0 || t >= (16 >> level)) return;

		ImageWrap iw = ImageWrap::Black;

		(pyramid[level])(s, t) = 0.25f *
			(texel(2 * s, 2 * t, level - 1, pyramid, iw) + texel(2 * s + 1, 2 * t, level - 1, pyramid, iw) +
				texel(2 * s, 2 * t + 1, level - 1, pyramid, iw) + texel(2 * s + 1, 2 * t + 1, level - 1, pyramid, iw));
	}

}

#endif