#ifndef __mipmap_kernel_cu__
#define __mipmap_kernel_cu__

#include "Mipmap.cuh"
#include "../../core/ImageI.cuh"

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

	__host__ Mipmap CreateMipMap(const char* file, ImageWrap wrapMode) {

		int width, height;
		Spectrum* img = loadImageFile_s(file, &width, &height);
		Spectrum* d_img = passToDevice(img, width * height);

		int level = log2f(fmaxf(width, height));

		BlockedArray<Spectrum>* d_pyr;
		gpuErrCheck(cudaMalloc((void**)&d_pyr, level * sizeof(BlockedArray<Spectrum>)));

		dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);
		dim3 grid;
		int w = width, h = height;

		for (int l = 0; l < level; l++) {

			if (l == 0) {
				pyramidBaseInit << <1, 1 >> > (d_pyr, width, height, d_img, level);
			}
			else {

				grid = dim3(w / BLOCK_SIZE + (w % BLOCK_SIZE != 0),
					h / BLOCK_SIZE + (h % BLOCK_SIZE != 0));

				pyramidInitKernel << <grid, block >> > (d_pyr, l, wrapMode);
			}
			gpuErrCheck(cudaDeviceSynchronize());
			gpuErrCheck(cudaPeekAtLastError());
			w = fmaxf(1, w / 2);
			h = fmaxf(1, h / 2);
		}

		return Mipmap(wrapMode, d_pyr, width, height, level);
	}
}

#endif