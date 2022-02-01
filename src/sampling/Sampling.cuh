#ifndef __sampling_cuh__
#define __sampling_cuh__

#include "../geometry/Geometry.cuh"

namespace lts {

	__device__ inline Vector3f uniformSampleHemisphere(const Point2f& omega) {
		float z = omega[0];
		float r = sqrtf(fmaxf(0, 1.0f - z * z));
		float phi = 2 * M_PI * omega[1];

		return Vector3f(r * cosf(phi), r * sinf(phi), z);
	}

	__device__ inline float uniformSampleHemispherePDF() {
		return 0.5f * M_1_PI;
	}

	__device__ inline Vector3f uniformSampleSphere(const Point2f& omega) {
		float z = 1 - 2 * omega[0];
		float r = sqrtf(fmaxf(0, 1.0f - z * z));
		float phi = 2 * M_PI * omega[1];

		return Vector3f(r * cosf(phi), r * sinf(phi), z);
	}

	__device__ inline float uniformSampleSpherePDF() {
		return 0.25f * M_1_PI;
	}

	__device__ inline Point2f uniformSampleDisk(const Point2f& omega) {
		float r = sqrtf(omega[0]);
		float theta = 2 * M_PI * omega[1];
		return Point2f(r * cosf(theta), r * sinf(theta));
	}

	__device__ inline float uniformSampleDiskPDF() {
		return M_1_PI;
	}

	// More likely to sample at the apex of the hemisphere
	__device__ inline Vector3f cosineSampleHemisphere(const Point2f& omega) {

		Point2f d = uniformSampleDisk(omega);
		float z = sqrtf(fmaxf(0.0f, 1 - d.x * d.x - d.y * d.y));

		return Vector3f(d.x, d.y, z);
	}

	__device__ inline float cosineSampleHemispherePDF(float cosTheta) {
		return cosTheta * M_1_PI;
	}

	__device__ inline Point2f uniformSampleTriangle(const Point2f& omega) {

		float t = sqrtf(omega[0]);

		return Point2f(1 - t, omega[1] * t);
	}

	/**
	* Heuristic used in multiple importance sampling to lower the variance of a product of two functions.
	* @param nf and ng - the number of samples of the respective f and g functions
	* @param fPdf and gPdf - the probability density of the respective f and g functions
	* @return w, the power heuristic which is used to lower variance
	*/
	__device__ inline float powerHeuristic(int nf, float fPdf, int ng, float gPdf) {
		float f = nf * fPdf, g = ng * gPdf;
		return (f * f) / (f * f + g * g);
	}


	__device__ inline int findDistribInterval(int size, float* cdf, float u) {

		int first = 0, len = size;

		while (len > 0) {
			int half = len >> 1, middle = first + half;

			if (cdf[middle] <= u) {
				first = middle + 1;
				len -= half + 1;
			}
			else len = half;
		}

		return clamp(first - 1, 0, size - 2);
	}

	struct Distribution1D {

		float funcInt;
		int size;
		float* func, * cdf;

		__device__ Distribution1D(const float* f, int n) : size(n) {

			func = new float[size];
			for (int i = 0; i < size; i++) {
				func[i] = f[i];
			}

			cdf = new float[size + 1];
			cdf[0] = 0.0f;
			for (int i = 1; i < size + 1; i++) {
				cdf[i] = cdf[i - 1] + func[i - 1] / size;
			}

			funcInt = cdf[size];

			if (funcInt == 0) {
				for (int i = 1; i < size + 1; i++) cdf[i] = (float)i / (float)n;
			}
			else {
				for (int i = 1; i < size + 1; i++) cdf[i] /= funcInt;
			}
		}

		//__device__ ~Distribution1D() { free(func); free(cdf); }

		__device__ int findInterval(float u) const {

			int first = 0, len = size;

			while (len > 0) {
				int half = len >> 1, middle = first + half;

				if (cdf[middle] <= u) {
					first = middle + 1;
					len -= half + 1;
				}
				else len = half;
			}

			return clamp(first - 1, 0, size - 2);
		}

		__device__ float sampleContinous(float u, float* pdf, int* off = nullptr) const {

			int offset = findDistribInterval(size + 1, cdf, u);

			if (off) *off = offset;

			float du = u - cdf[offset];
			if ((cdf[offset + 1] - cdf[offset]) > 0)
				du /= (cdf[offset + 1] - cdf[offset]);

			if (pdf) *pdf = func[offset] / funcInt;
			return (offset + du) / size;
		}

		__device__ int sampleDiscrete(float u, float* pdf = nullptr, float* uRemapped = nullptr) const {

			int offset = findInterval(u);

			if (pdf) *pdf = func[offset] / (funcInt * size);
			if (uRemapped) *uRemapped = (u - cdf[offset]) / (cdf[offset + 1] - cdf[offset]);
			return offset;
		}

		__device__ float discretePdf(int index) const {
			return func[index] / (funcInt * size);
		}
	};
}

#endif