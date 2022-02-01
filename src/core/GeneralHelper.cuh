#ifndef __general_helper_cuh__
#define __general_helper_cuh__

#include "ErrorHelper.cuh"

namespace lts {

#ifndef M_PI
#define M_PI 3.1415926535898f // pi
#endif

#ifndef M_1_PI
#define M_1_PI 0.318309886183790671538f // 1 / pi
#endif

#ifndef M_1_SQRT_PI
#define M_1_SQRT_PI 0.56418958354f // 1 / sqrt(pi)
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif

#ifdef __INTELLISENSE__
#include "IntellisenseCudaIntrinsics.h"
#endif

	template <typename T>
	__host__ inline T* passToDevice(const T* host_ptr, int count = 1) {
		T* device_ptr;
		gpuErrCheck(cudaMalloc((void**)&device_ptr, count * sizeof(T)));
		gpuErrCheck(cudaMemcpy(device_ptr, host_ptr, count * sizeof(T), cudaMemcpyHostToDevice));
		return device_ptr;
	}

	__host__ __device__ inline void swap(float& a, float& b) {
		float c = a;
		a = b;
		b = c;
	}
	__host__ __device__ inline float linearInterpolation(float t, float v1, float v2) { return (1 - t) * v1 + t * v2; }
	__host__ __device__ inline float toRadians(float deg) {
		return deg * (M_PI / 180);
	}
	__host__ __device__ inline float clamp(float f, float min, float max) {
		return (f < min) ? min : ((f > max) ? max : f);
	}
	__host__ __device__ inline float clamp(float f) {
		return (f < 0) ? 0 : ((f > 1) ? 1 : f);
	}

	/**
	* Estimation of the inverse of the error function.
	* @param x - input
	*/
	__host__ __device__ inline float errfInv(float x) {
		float w, p;
		x = clamp(x, -.99999f, .99999f);
		w = -logf((1 - x) * (1 + x));
		if (w < 5) {
			w = w - 2.5f;
			p = 2.81022636e-08f;
			p = 3.43273939e-07f + p * w;
			p = -3.5233877e-06f + p * w;
			p = -4.39150654e-06f + p * w;
			p = 0.00021858087f + p * w;
			p = -0.00125372503f + p * w;
			p = -0.00417768164f + p * w;
			p = 0.246640727f + p * w;
			p = 1.50140941f + p * w;
		}
		else {
			w = sqrtf(w) - 3;
			p = -0.000200214257f;
			p = 0.000100950558f + p * w;
			p = 0.00134934322f + p * w;
			p = -0.00367342844f + p * w;
			p = 0.00573950773f + p * w;
			p = -0.0076224613f + p * w;
			p = 0.00943887047f + p * w;
			p = 1.00167406f + p * w;
			p = 2.83297682f + p * w;
		}
		return p * x;
	}

	/**
	* Estimation of the error function.
	* @param x - input
	*/
	__host__ __device__ inline float errf(float x) {
		// constants
		float a1 = 0.254829592f;
		float a2 = -0.284496736f;
		float a3 = 1.421413741f;
		float a4 = -1.453152027f;
		float a5 = 1.061405429f;
		float p = 0.3275911f;

		// Save the sign of x
		int sign = 1;
		if (x < 0) sign = -1;
		x = fabsf(x);

		// A&S formula 7.1.26
		float t = 1 / (1 + p * x);
		float y =
			1 -
			(((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * expf(-x * x);

		return sign * y;
	}

	__host__ __device__ inline bool solveLinearSystem2x2(const float a[2][2], const float b[2], float* x0, float* x1) {
		float det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
		if (fabsf(det) < 1e-10f) return false;
		*x0 = (a[1][1] * b[0] - a[0][1] * b[1]) / det;
		*x1 = (a[0][0] * b[1] - a[1][0] * b[0]) / det;
		if (isnan(*x0) || isnan(*x1)) return false;
		return true;
	}

	template <typename T>
	__host__ __device__ inline bool isNaN(const T x) { return isnan(x); }
	template <>
	__host__ __device__ inline bool isNaN(const int x) { return false; }

	__device__ inline unsigned int wangHash(unsigned int a) {
		a = (a ^ 61) ^ (a >> 16);
		a = a + (a << 3);
		a = a ^ (a >> 4);
		a = a * 0x27d4eb2d;
		a = a ^ (a >> 15);
		return a;
	}

	__device__ __forceinline__ float fatomicMin(float* addr, float value) {
		return (value >= 0) ? atomicMin((int*)addr, __float_as_int(value)) :
			atomicMax((unsigned int*)addr, __float_as_uint(value));
	}

	__device__ __forceinline__ float fatomicMax(float* addr, float value) {

		return (value >= 0) ? atomicMax((int*)addr, __float_as_int(value)) :
			atomicMin((unsigned int*)addr, __float_as_uint(value));
	}
}

#endif