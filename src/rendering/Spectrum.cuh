#ifndef __spectrum_cuh__
#define __spectrum_cuh__

#include "../core/GeneralHelper.cuh"

namespace lts {

	/**
	* Class for colors in spectral forms. Is either in RGB format or in XYZ format.
	*/
	class Spectrum {

	public:

		float r, g, b;

		__host__ __device__ Spectrum() : r(0), g(0), b(0) {}

		__host__ __device__ Spectrum(float v) : r(v), g(v), b(v) {}

		__host__ __device__ Spectrum(float r, float g, float b) : r(r), g(g), b(b) {}

		__host__ __device__ Spectrum(float rgb[3]) : r(rgb[0]), g(rgb[1]), b(rgb[2]) {}

		__host__ __device__ Spectrum& operator+=(const Spectrum& s) {
			r += s.r;
			g += s.g;
			b += s.b;
			return *this;
		}

		__host__ __device__ Spectrum operator+(const Spectrum& s) const {

			Spectrum ret;

			ret.r = r + s.r;
			ret.g = g + s.g;
			ret.b = b + s.b;
			return ret;
		}

		/**
		* Atomically add another spectrum to this one.
		* @param s - Spectrum to add.
		*/
		__device__ Spectrum& atomicAddition(const Spectrum& s) {

			atomicAdd(&r, s.r);
			atomicAdd(&g, s.g);
			atomicAdd(&b, s.b);

			return *this;
		}

		__host__ __device__ Spectrum& operator-=(const Spectrum& s) {

			r -= s.r;
			g -= s.g;
			b -= s.b;

			return *this;
		}

		__host__ __device__ Spectrum operator-(const Spectrum& s) const {

			Spectrum ret;

			ret.r = r - s.r;
			ret.g = g - s.g;
			ret.b = b - s.b;

			return ret;
		}

		__host__ __device__ Spectrum& operator*=(const Spectrum& s) {

			r *= s.r;
			g *= s.g;
			b *= s.b;

			return *this;
		}

		__host__ __device__ Spectrum& operator*=(float v) {

			r *= v;
			g *= v;
			b *= v;

			return *this;
		}

		__host__ __device__ Spectrum operator*(const Spectrum& s) const {

			Spectrum ret;

			ret.r = r * s.r;
			ret.g = g * s.g;
			ret.b = b * s.b;

			return ret;
		}

		__host__ __device__ Spectrum operator*(float v) const {

			Spectrum ret;

			ret.r = r * v;
			ret.g = g * v;
			ret.b = b * v;

			return ret;
		}

		__host__ __device__ friend inline Spectrum operator*(float v, const Spectrum& s) {

			return s * v;
		}

		__host__ __device__ Spectrum& operator/=(const Spectrum& s) {

			r /= s.r;
			g /= s.g;
			b /= s.b;

			assert(!hasNaNs());

			return *this;
		}

		__host__ __device__ Spectrum& operator/=(float v) {

			assert(v != 0);

			r /= v;
			g /= v;
			b /= v;

			return *this;
		}

		__host__ __device__ Spectrum operator/(const Spectrum& s) const {

			Spectrum ret;

			ret.r = r / s.r;
			ret.g = g / s.g;
			ret.b = b / s.b;

			assert(!ret.hasNaNs());

			return ret;
		}

		__host__ __device__ Spectrum operator/(float v) const {

			Spectrum ret;

			assert(v != 0);

			ret.r = r / v;
			ret.g = g / v;
			ret.b = b / v;

			return ret;
		}

		__host__ __device__ Spectrum operator-() {

			return *this * -1;
		}

		/**
		* Clamps all the channels of the spectrum between two values.
		* @param low - minimum value
		* @param high - maximum value
		*/
		__host__ __device__ Spectrum clamp(float low = 0.0f, float high = INFINITY) const {

			Spectrum ret;

			ret.r = lts::clamp(r, low, high);
			ret.g = lts::clamp(g, low, high);
			ret.b = lts::clamp(b, low, high);

			return ret;
		}

		__host__ __device__ bool operator==(const Spectrum& s) const {

			if (r != s.r) return false;
			if (g != s.g) return false;
			if (b != s.b) return false;

			return true;
		}

		__host__ __device__ bool operator!=(const Spectrum& s) const {
			return !(*this == s);
		}

		/**
		* Tests if the spectrum is the color black.
		*/
		__host__ __device__ bool isBlack() const {

			if (r != 0.0f) return false;
			if (g != 0.0f) return false;
			if (b != 0.0f) return false;

			return true;
		}

		/**
		* Compute the luminance of the spectrum.
		*/
		__host__ __device__ float y() const {
			const float YWeight[3] = { 0.212671f, 0.715160f, 0.072169f };
			return YWeight[0] * r + YWeight[1] * g + YWeight[2] * b;
		}

		__host__ __device__ bool hasNaNs() {
			return isNaN(r) || isNaN(g) || isNaN(b);
		}
	};

	/**
	* Compute the sqrt of all the values in the spectrum
	* @param s - Initial spectrum
	* @return Sqrt of s
	*/
	__host__ __device__ inline Spectrum sqrt(const Spectrum& s) {
		Spectrum ret;

		ret.r = sqrtf(s.r);
		ret.g = sqrtf(s.g);
		ret.b = sqrtf(s.b);

		assert(!ret.hasNaNs());
		return ret;
	}

	/**
	* Interpolation between two spectrums.
	* @param t - Interpolation constant
	* @param s1 - First spectrum
	* @param s2 - Second spectrum
	* @param Spectrum between s1 and s2.
	*/
	__host__ __device__ inline Spectrum linearInterpolation(float t, const Spectrum& s1, const Spectrum& s2) {
		return (1 - t) * s1 + t * s2;
	}

	/**
	* Transform a luminance spectrum to a RGB spectrum.
	* @param xyz - Luminance spectrum
	* @param rgb - Return value in RGB
	*/
	__host__ __device__ inline void XYZtoRGB(const float xyz[3], float rgb[3]) {
		rgb[0] = 3.240479f * xyz[0] - 1.537150f * xyz[1] - 0.498535f * xyz[2];
		rgb[1] = -0.969256f * xyz[0] + 1.875991f * xyz[1] + 0.041556f * xyz[2];
		rgb[2] = 0.055648f * xyz[0] - 0.204043f * xyz[1] + 1.057311f * xyz[2];
	}

	/**
	* Transform a RGB spectrum to a luminance spectrum.
	* @param xyz - RGB spectrum
	* @param rgb - Return value in luminance
	*/
	__host__ __device__ inline void RGBtoXYZ(const float rgb[3], float xyz[3]) {
		xyz[0] = 0.412453f * rgb[0] + 0.357580f * rgb[1] + 0.180423f * rgb[2];
		xyz[1] = 0.212671f * rgb[0] + 0.715160f * rgb[1] + 0.072169f * rgb[2];
		xyz[2] = 0.019334f * rgb[0] + 0.119193f * rgb[1] + 0.950227f * rgb[2];
	}
}

#endif