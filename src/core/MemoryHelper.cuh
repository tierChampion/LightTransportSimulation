#ifndef __memory_cuh__
#define __memory_cuh__

#include "GeneralHelper.cuh"

namespace lts {

#ifndef L1_CACHE_SIZE
#define L1_CACHE_SIZE 64
#endif

#ifndef LOG_BLOCK_SIZE
#define LOG_BLOCK_SIZE 2
#endif

	__device__ inline void* allocAligned(size_t size) {

		return _aligned_malloc(size, L1_CACHE_SIZE);
	}

	template <typename T>
	__device__ inline T* allocAligned(size_t count) {
		return (T*)allocAligned(count * sizeof(T));
	}

	__device__ inline void freeAligned(void* block) {

		return _aligned_free(block);
	}

	// 2d array stored in squares instead of rows
	template <typename T>
	class BlockedArray {

		T* data;
		const int uRes, vRes, uBlocks;

	public:

		__host__ __device__ BlockedArray(int uRes, int vRes) : uRes(uRes), vRes(vRes), uBlocks(uRes >> LOG_BLOCK_SIZE)
		{}

		__device__ BlockedArray(int uRes, int vRes, const T* d)
			: uRes(uRes), vRes(vRes), uBlocks(uRes >> LOG_BLOCK_SIZE)
		{
			int nAlloc = roundUp(uRes) * roundUp(vRes);
			data = allocAligned<T>(nAlloc);
			for (int i = 0; i < nAlloc; i++) {
				new (&data[i]) T();
			}
			if (d) {
				for (int v = 0; v < vRes; v++) {
					for (int u = 0; u < uRes; u++) {
						(*this)(u, v) = d[v * uRes + u];
					}
				}
			}
		}
		__device__ int blockSize() const { return 1 << LOG_BLOCK_SIZE; }
		__device__ int roundUp(int x) const {
			return (x + blockSize() - 1) & ~(blockSize() - 1);
		}

		__device__ ~BlockedArray() {
			for (int i = 0; i < uRes * vRes; i++) data[i].~T();

			freeAligned(data);
		}

		__device__ int getBlock(int a) const { return a >> LOG_BLOCK_SIZE; }
		__device__ int offset(int a) const { return (a & (blockSize() - 1)); }

		__device__ T& operator()(int u, int v) {
			int bu = block(u), bv = block(v);
			int ou = offset(u), ov = block(v);
			int offset = blockSize() * blockSize() * (uBlocks * bv + bu);
			offset += blockSize() * ov + ou;
			return data[offset];
		}

		__device__ const T& operator()(int u, int v) {
			int bu = block(u), bv = block(v);
			int ou = offset(u), ov = block(v);
			int offset = blockSize() * blockSize() * (uBlocks * bv + bu);
			offset += blockSize() * ov + ou;
			return data[offset];
		}

		__device__ void getLinearArray(T* arr) const {
			for (int v = 0; v < vRes; v++) {
				for (int u = 0; u < uRes; u++) {
					*arr++ = (*this)(u, v);
				}
			}
		}
	};
}

#endif