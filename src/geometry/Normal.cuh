#ifndef __normal_cuh__
#define __normal_cuh__

#include "Vector.cuh"

namespace lts {

	template <typename T> class Normal3 {

	public:

		T x, y, z;

		__host__ __device__ Normal3() { x = y = z = 0; }
		__host__ __device__ Normal3(T xx, T yy, T zz) : x(xx), y(yy), z(zz) { assert(!hasNaNs()); }

		__host__ __device__ explicit Normal3<T>(const Vector3<T>& v) : x(v.x), y(v.y), z(v.z) { assert(!v.hasNaNs()); }
		__host__ __device__ Normal3<T> operator+(const Normal3<T>& n) const {
			assert(!n.hasNaNs());
			return Normal3<T>(x + n.x, y + n.y, z + n.z);
		}

		__host__ __device__ Normal3<T> operator-(const Normal3<T>& n) const {
			assert(!n.hasNaNs());
			return Normal3<T>(x - n.x, y - n.y, z - n.z);
		}

		__host__ __device__ Normal3<T>& operator+=(const Normal3<T>& n) {
			assert(!n.hasNaNs());
			x += n.x;
			y += n.y;
			z += n.z;
			return *this;
		}

		__host__ __device__ Normal3<T>& operator-=(const Normal3<T>& n) {
			assert(!n.hasNaNs());
			x -= n.x;
			y -= n.y;
			z -= n.z;
			return *this;
		}



		template <typename U>
		__host__ __device__ Normal3<T> operator*(U f) const {
			return Normal3<T>(f * x, f * y, f * z);
		}

		template <typename U>
		__host__ __device__ Normal3<T> operator/(U f) const {
			assert(f != 0);
			float inv = (float)1 / f;
			return Normal3<T>(x * inv, y * inv, z * inv);
		}

		template <typename U>
		__host__ __device__ Normal3<T>& operator*=(U f) const {
			x *= f;
			y *= f;
			z *= f;
			return *this;
		}

		template <typename U>
		__host__ __device__ Normal3<T>& operator/=(U f) const {
			assert(f != 0);
			float inv = (float)1 / f;
			x *= inv;
			y *= inv;
			z *= inv;
			return *this;
		}

		__host__ __device__ Normal3<T> operator-() const { return Normal3<T>(-x, -y, -z); }

		__host__ __device__ T operator[](int i) const {
			assert(i >= 0 && i <= 2);
			if (i == 0) return x;
			if (i == 1) return y;
			return z;
		}

		__host__ __device__ T& operator[](int i) {
			assert(i >= 0 && i <= 2);
			if (i == 0) return x;
			if (i == 1) return y;
			return z;
		}

		__host__ __device__ bool operator==(const Normal3<T>& n) const { return x == n.x && y == n.y && z == n.z; }
		__host__ __device__ bool operator!=(const Normal3<T>& n) const { return x != n.x || y != n.y || z != n.z; }

		__host__ __device__ T lengthSquared() const { return x * x + y * y + z * z; }
		__host__ __device__ float length() const { return (float)sqrtf(lengthSquared()); }

		__host__ __device__ void print() const { printf("Normal3 -> x: %f y: %f z: %f\n", (float)x, (float)y, (float)z); }
		__host__ __device__ bool hasNaNs() const { return isNaN(x) || isNaN(y) || isNaN(z); }
	};

	// NORMAL TYPE
	typedef Normal3<float> Normal3f;

}

#endif