#ifndef __error_helper_cuh__
#define __error_helper_cuh__

#include <iostream>
#include <chrono>
#include <assert.h>
// CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <device_atomic_functions.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
// CURAND
#include <curand.h>
#include <curand_kernel.h>

namespace lts {

	/**
		* Checks errors with cuda and gpu calls.
		* @param cuda function that needs to be checked for errors, the file and the line.
		*/
#define gpuErrCheck(ans) {gpuAssert((ans), __FILE__, __LINE__);}
	inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
		if (code != cudaSuccess) {
			fprintf(stderr, "GPU assertion failed: %s %s %d\n", cudaGetErrorString(code), file, line);
			if (abort) exit(code);
		}
	}

	/// Acceptable error range for 32-bit floating values
#ifndef FLT_EPSILON
#define FLT_EPSILON 1.19209e-07
#endif
	/// Machine epsilon for shadow rays. Need to be less precise
#ifndef SHADOW_EPSILON
#define SHADOW_EPSILON 0.0001f
#endif

	__device__ inline uint32_t floatToBits(float f) {
		uint32_t ui;
		memcpy(&ui, &f, sizeof(float));
		return ui;
	}

	__device__ inline float bitsToFloat(uint32_t ui) {
		float f;
		memcpy(&f, &ui, sizeof(uint32_t));
		return f;
	}

	__device__ inline float nextFloatUp(float f) {
		if (isinf(f) && f > 0.0f)
			return f;
		if (f == -0.0f) f = 0.0f;

		uint32_t ui = floatToBits(f);
		if (f >= 0) ui++;
		else ui--;
		return bitsToFloat(ui);
	}

	__device__ inline float nextFloatDown(float f) {
		if (isinf(f) && f > 0.0f)
			return f;
		if (f == -0.0f) f = 0.0f;

		uint32_t ui = floatToBits(f);
		if (f >= 0) ui--;
		else ui++;
		return bitsToFloat(ui);
	}

	__device__ static constexpr float machineEpsilon() { return 0.5f * FLT_EPSILON; }

	/**
	* Floating point error of n values.
	* @param n - Number of values to account for.
	*/
	__device__ inline float gamma(int n) {
		return (n * machineEpsilon()) / (1 - n * machineEpsilon());
	}

	class EFloat {

		float v;
		float high;
		float low;

		__device__ inline void check() const {
			if (!isinf(low) && !isnan(low) && !isinf(high) && !isnan(high)) {
				assert(low <= high);
			}
		}

	public:

		__device__ EFloat() {}
		__device__ EFloat(float v, float err = 0.0f) : v(v) {
			if (err == 0.0f) {
				low = high = v;
			}
			else {
				low = nextFloatDown(v - err);
				high = nextFloatUp(v + err);
			}
		}

		__device__ explicit operator float() const { return v; }

		__device__ float getAbsoluteError() const { return fmaxf(fabsf(high - v), fabsf(v - low)); }

		__device__ float upperBound() const { return high; }
		__device__ float lowerBound() const { return low; }

		__device__ EFloat operator+(EFloat fe) const {
			EFloat r;
			r.v = v + fe.v;
			r.low = nextFloatDown(lowerBound() + fe.lowerBound());
			r.high = nextFloatUp(upperBound() + fe.upperBound());
			r.check();
			return r;
		}

		__device__ EFloat operator-(EFloat fe) const {
			EFloat r;
			r.v = v - fe.v;
			r.low = nextFloatDown(lowerBound() - fe.lowerBound());
			r.high = nextFloatUp(upperBound() - fe.lowerBound());
			r.check();
			return r;
		}

		__device__ EFloat operator*(EFloat fe) const {
			EFloat r;
			r.v = v * fe.v;

			float prod[4] = {
				lowerBound() * fe.lowerBound(), upperBound() * fe.lowerBound(),
				lowerBound() * fe.upperBound(), upperBound() * fe.upperBound()
			};

			r.low = nextFloatDown(fminf(fminf(prod[0], prod[1]), fminf(prod[2], prod[3])));
			r.high = nextFloatUp(fmaxf(fmaxf(prod[0], prod[1]), fmaxf(prod[2], prod[3])));
			r.check();
			return r;
		}

		__device__ EFloat operator/(EFloat fe) const {
			EFloat r;
			r.v = v / fe.v;

			if (fe.low < 0 && fe.high > 0) {
				r.low = -INFINITY;
				r.high = INFINITY;
			}
			else {

				float div[4] = {
				lowerBound() / fe.lowerBound(), upperBound() / fe.lowerBound(),
				lowerBound() / fe.upperBound(), upperBound() / fe.upperBound()
				};

				r.low = nextFloatDown(fminf(fminf(div[0], div[1]), fminf(div[2], div[3])));
				r.high = nextFloatUp(fmaxf(fmaxf(div[0], div[1]), fmaxf(div[2], div[3])));
			}
			r.check();
			return r;
		}

		__device__ EFloat operator-() const {
			EFloat r;
			r.v = -r.v;

			r.low = -high;
			r.high = -low;
			r.check();
			return r;
		}

		__device__ inline bool operator==(EFloat fe) const { return v == fe.v; }

		__device__ EFloat(const EFloat& fe) {
			fe.check();
			v = fe.v;
			low = fe.low;
			high = fe.high;
		}

		__device__ EFloat& operator=(const EFloat& fe) {
			fe.check();
			if (&fe != this) {
				v = fe.v;
				low = fe.low;
				high = fe.high;
			}

			return *this;
		}

		__device__ void print() {
			printf("EFloat -> v: %f low: %f high: %f\n", v, low, high);
		}

		__device__ friend inline EFloat eSqrt(EFloat fe);
		__device__ friend inline EFloat eAbs(EFloat fe);
		__device__ friend inline bool eQuadratic(EFloat a, EFloat b, EFloat c, EFloat* x1, EFloat* x2);
	};

	__device__ inline EFloat operator*(float f, EFloat fe) { return EFloat(f) * fe; }
	__device__ inline EFloat operator/(float f, EFloat fe) { return EFloat(f) / fe; }
	__device__ inline EFloat operator+(float f, EFloat fe) { return EFloat(f) + fe; }
	__device__ inline EFloat operator-(float f, EFloat fe) { return EFloat(f) - fe; }

	__device__ inline EFloat eSqrt(EFloat fe) {
		EFloat r;
		r.v = sqrtf(fe.v);
		r.low = nextFloatDown(sqrtf(fe.low));
		r.high = nextFloatUp(sqrtf(fe.high));
		r.check();
		return r;
	}

	__device__ inline EFloat eAbs(EFloat fe) {
		// Only positive
		if (fe.low >= 0) {
			return fe;
		}
		// Only negative
		else if (fe.high <= 0) {
			EFloat r;
			r.v = -fe.v;
			r.low = -fe.high;
			r.high = -fe.low;
			r.check();
			return r;
		}
		// Both positive and negative in interval
		else {
			EFloat r;
			r.v = fabsf(fe.v);
			r.low = 0;
			r.high = fmaxf(-fe.low, fe.high);
			r.check();
			return r;
		}
	}

	__device__ inline bool eQuadratic(EFloat a, EFloat b, EFloat c, EFloat* x0, EFloat* x1) {

		double discrim = (double)b.v * (double)b.v - 4.0 * (double)a.v * (double)c.v;
		if (discrim < 0.0) return false;
		double rootDiscrim = sqrtf(discrim);
		EFloat floatRootDiscrim(rootDiscrim, machineEpsilon() * rootDiscrim);

		EFloat q;
		if ((float)b < 0)
			q = -0.5f * (b - floatRootDiscrim);
		else
			q = -0.5f * (b + floatRootDiscrim);
		*x0 = q / a;
		*x1 = c / q;
		if ((float)*x0 > (float)*x1) {
			EFloat temp = *x0;
			*x0 = *x1;
			*x1 = temp;
		}

		return true;
	}
}

#endif