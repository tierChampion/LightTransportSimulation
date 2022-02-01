#ifndef __vector_cuh__
#define __vector_cuh__

#include "../core/GeneralHelper.cuh"

namespace lts {

	template <typename T> class Point2;
	template <typename T> class Point3;
	template <typename T> class Normal3;

	template <typename T> class Vector2 {

	public:

		T x, y;

		__host__ __device__ Vector2() { x = y = 0; }
		__host__ __device__ Vector2(T xx, T yy) : x(xx), y(yy) { assert(!hasNaNs()); }

		__host__ __device__ explicit Vector2(const Point2<T>& p) : x(p.x), y(p.y) { assert(!p.hasNaNs()); }
		__host__ __device__ explicit Vector2(const Point3<T>& p) : x(p.x), y(p.y) { assert(!p.hasNaNs()); }

		__host__ __device__ Vector2<T> operator+(const Vector2<T>& v) const {
			assert(!v.hasNaNs());
			return Vector2<T>(x + v.x, y + v.y);
		}

		__host__ __device__ Vector2<T> operator-(const Vector2<T>& v) const {
			assert(!v.hasNaNs());
			return Vector2<T>(x - v.x, y - v.y);
		}

		__host__ __device__ Vector2<T>& operator+=(const Vector2<T>& v) {
			assert(!v.hasNaNs());
			x += v.x;
			y += v.y;
			return *this;
		}

		__host__ __device__ Vector2<T>& operator-=(const Vector2<T>& v) {
			assert(!v.hasNaNs());
			x -= v.x;
			y -= v.y;
			return *this;
		}

		template <typename U>
		__host__ __device__ Vector2<T> operator*(U f) const {
			assert(!isNaN(f));
			return Vector2<T>(f * x, f * y);
		}

		template <typename U>
		__host__ __device__ Vector2<T> operator/(U f) const {
			assert(!isNaN(f));
			float inv = (float)1 / f;
			return Vector2<T>(x * inv, y * inv);
		}

		template <typename U>
		__host__ __device__ Vector2<T>& operator*=(U f) const {
			assert(!isNaN(f));
			x *= f;
			y *= f;
			return *this;
		}

		template <typename U>
		__host__ __device__ Vector2<T>& operator/=(U f) const {
			assert(!isNaN(f));
			float inv = (float)1 / f;
			x *= inv;
			y *= inv;
			return *this;
		}

		__host__ __device__ Vector2<T> operator-() const { return Vector2<T>(-x, -y); }

		__host__ __device__ T operator[](int i) const {
			assert(i >= 0 && i <= 1);
			if (i == 0) return x;
			return y;
		}

		__host__ __device__ T& operator[](int i) {
			assert(i >= 0 && i <= 1);
			if (i == 0) return x;
			return y;
		}

		__host__ __device__ bool operator==(const Vector2<T>& v) const { return x == v.x && y == v.y; }
		__host__ __device__ bool operator!=(const Vector2<T>& v) const { return x != v.x || y != v.y; }

		__host__ __device__ T lengthSquared() const { return x * x + y * y; }
		__host__ __device__ float length() const { return (float)sqrtf(lengthSquared()); }

		__host__ __device__ void print() const { printf("Vector2 -> x: %f y: %f\n", (float)x, (float)y); }
		__host__ __device__ bool hasNaNs() const { return isNaN(x) || isNaN(y); }
	};

	template <typename T> class Vector3 {

	public:

		T x, y, z;

		__host__ __device__ Vector3() { x = y = z = 0; }
		__host__ __device__ Vector3(T xx, T yy, T zz) : x(xx), y(yy), z(zz) { assert(!hasNaNs()); }

		__host__ __device__ explicit Vector3(const Point3<T>& p) : x(p.x), y(p.y), z(p.z) { assert(!p.hasNaNs()); }
		__host__ __device__ explicit Vector3(const Normal3<T>& n) : x(n.x), y(n.y), z(n.z) { assert(!n.hasNaNs()); }

		__host__ __device__ Vector3<T> operator+(const Vector3<T>& v) const {
			assert(!v.hasNaNs());
			return Vector3<T>(x + v.x, y + v.y, z + v.z);
		}

		__host__ __device__ Vector3<T> operator-(const Vector3<T>& v) const {
			assert(!v.hasNaNs());
			return Vector3<T>(x - v.x, y - v.y, z - v.z);
		}

		__host__ __device__ Vector3<T>& operator+=(const Vector3<T>& v) {
			assert(!v.hasNaNs());
			x += v.x;
			y += v.y;
			z += v.z;
			return *this;
		}

		__host__ __device__ Vector3<T>& operator-=(const Vector3<T>& v) {
			assert(!v.hasNaNs());
			x -= v.x;
			y -= v.y;
			z -= v.z;
			return *this;
		}

		template <typename U>
		__host__ __device__ Vector3<T> operator*(U f) const {
			return Vector3<T>(f * x, f * y, f * z);
		}

		template <typename U>
		__host__ __device__ Vector3<T> operator/(U f) const {
			assert(f != 0);
			float inv = (float)1 / f;
			return Vector3<T>(x * inv, y * inv, z * inv);
		}

		template <typename U>
		__host__ __device__ Vector3<T>& operator*=(U f) {
			x *= f;
			y *= f;
			z *= f;
			return *this;
		}

		template <typename U>
		__host__ __device__ Vector3<T>& operator/=(U f) {
			assert(f != 0);
			float inv = (float)1 / f;
			x *= inv;
			y *= inv;
			z *= inv;
			return *this;
		}

		__host__ __device__ Vector3<T> operator-() const { return Vector3<T>(-x, -y, -z); }

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

		__host__ __device__ bool operator==(const Vector3<T>& v) const { return x == v.x && y == v.y && z == v.z; }
		__host__ __device__ bool operator!=(const Vector3<T>& v) const { return x != v.x || y != v.y || z != v.z; }

		__host__ __device__ T lengthSquared() const { return x * x + y * y + z * z; }
		__host__ __device__ float length() const { return (float)sqrtf(lengthSquared()); }

		__host__ __device__ void print() const { printf("Vector3 -> x: %f y: %f z: %f\n", (float)x, (float)y, (float)z); }
		__host__ __device__ bool hasNaNs() const { return isNaN(x) || isNaN(y) || isNaN(z); }
	};

	// VECTOR TYPES
	typedef Vector2<float> Vector2f;
	typedef Vector3<float> Vector3f;
	typedef Vector2<EFloat> Vector2e;
	typedef Vector3<EFloat> Vector3e;
	typedef Vector2<int> Vector2i;
	typedef Vector3<int> Vector3i;
}

#endif