#ifndef __point_cuh__
#define __point_cuh__

#include "Vector.cuh"

namespace lts {

	template <typename T> class Point3;

	template <typename T> class Point2 {

	public:

		T x, y;

		__host__ __device__ Point2() { x = y = 0; }
		__host__ __device__ Point2(T xx, T yy) : x(xx), y(yy) { assert(!hasNaNs()); }

		__host__ __device__ explicit Point2(const Point3<T>& p) : x(p.x), y(p.y) { assert(!p.HasNaNs()); }

		template <typename U>
		__host__ __device__ explicit Point2(const Point2<U>& p) {
			x = (T)p.x;
			y = (T)p.y;
			assert(!hasNaNs());
		}

		template <typename U>
		__host__ __device__ explicit Point2(const Vector2<U>& v) {
			x = (T)v.x;
			y = (T)v.y;
			assert(!hasNaNs());
		}

		template <typename U>
		__host__ __device__ explicit operator Vector2<U>() const {
			return Vector2<U>(x, y);
		}

		__host__ __device__ Point2<T> operator+(const Vector2<T>& v) const {
			assert(!v.hasNaNs());
			return Point2<T>(x + v.x, y + v.y);
		}

		__host__ __device__ Point2<T> operator+(const Point2<T>& p) const {
			assert(!p.hasNaNs());
			return Point2<T>(x + p.x, y + p.y);
		}

		__host__ __device__ Vector2<T> operator-(const Point2<T>& p) const {
			assert(!p.hasNaNs());
			return Vector2<T>(x - p.x, y - p.y);
		}

		__host__ __device__ Point2<T> operator-(const Vector2<T>& v) const {
			assert(!v.hasNaNs());
			return Point2<T>(x - v.x, y - v.y);
		}

		__host__ __device__ Point2<T>& operator+=(const Point2<T>& p) {
			assert(!p.hasNaNs());
			x += p.x;
			y += p.y;
			return *this;
		}

		__host__ __device__ Point2<T>& operator-=(const Vector2<T>& v) {
			assert(!v.hasNaNs());
			x -= v.x;
			y -= v.y;
			return *this;
		}

		template <typename U>
		__host__ __device__ Point2<T> operator*(Vector2<U> v) const {
			assert(!v.hasNaNs());
			return Point2<T>(x * v.x, y * v.y);
		}

		template <typename U>
		__host__ __device__ Point2<T> operator*(U f) const {
			assert(!isNaN(f));
			return Point2<T>(f * x, f * y);
		}

		template <typename U>
		__host__ __device__ Point2<T> operator/(U f) const {
			assert(!isNaN(f));
			float inv = (float)1 / f;
			return Point2<T>(x * inv, y * inv);
		}

		template <typename U>
		__host__ __device__ Point2<T>& operator*=(U f) {
			assert(!isNaN(f));
			x *= f;
			y *= f;
			return *this;
		}

		template <typename U>
		__host__ __device__ Point2<T>& operator/=(U f) {
			assert(!isNaN(f));
			float inv = (float)1 / f;
			x *= inv;
			y *= inv;
			return *this;
		}

		__host__ __device__ Point2<T> operator-() const { return Point2<T>(-x, -y); }

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

		__host__ __device__ bool operator==(const Point2<T>& p) const { return x == p.x && y == p.y; }
		__host__ __device__ bool operator!=(const Point2<T>& p) const { return x != p.x || y != p.y; }

		__host__ __device__ void print() const { printf("Point2 -> x: %f y: %f\n", (float)x, (float)y); }
		__host__ __device__ bool hasNaNs() const { return isNaN(x) || isNaN(y); }
	};

	template <typename T>
	class Point3 {

	public:

		T x, y, z;

		__host__ __device__ Point3() { x = y = z = 0; }
		__host__ __device__ Point3(T xx, T yy, T zz) : x(xx), y(yy), z(zz) { assert(!hasNaNs()); }

		template <typename U>
		__host__ __device__ explicit Point3(const Point3<U>& p) {
			x = (T)p.x;
			y = (T)p.y;
			z = (T)p.z;
			assert(!hasNaNs());
		}

		template <typename U>
		__host__ __device__ explicit Point3(const Vector3<U>& v) {
			x = (T)v.x;
			y = (T)v.y;
			z = (T)v.z;
			assert(!hasNaNs());
		}

		template <typename U>
		__host__ __device__ explicit operator Vector3<U>() const {
			return Vector3<U>(x, y, z);
		}

		__host__ __device__ Point3<T> operator+(const Vector3<T>& v) const {
			assert(!v.hasNaNs());
			return Point3<T>(x + v.x, y + v.y, z + v.z);
		}

		__host__ __device__ Point3<T> operator+(const Point3<T>& p) const {
			assert(!p.hasNaNs());
			return Point3<T>(x + p.x, y + p.y, z + p.z);
		}

		__host__ __device__ Vector3<T> operator-(const Point3<T>& p) const {
			assert(!p.hasNaNs());
			return Vector3<T>(x - p.x, y - p.y, z - p.z);
		}

		__host__ __device__ Point3<T> operator-(const Vector3<T>& v) const {
			assert(!v.hasNaNs());
			return Point3<T>(x - v.x, y - v.y, z - v.z);
		}

		__host__ __device__ Point3<T>& operator+=(const Point3<T>& p) {
			assert(!p.hasNaNs());
			x += p.x;
			y += p.y;
			z += p.z;
			return *this;
		}

		__host__ __device__ Point3<T>& operator+=(const Vector3<T>& v) {
			assert(!v.hasNaNs());
			x += v.x;
			y += v.y;
			z += v.z;
			return *this;
		}

		__host__ __device__ Point3<T>& operator-=(const Vector3<T>& v) {
			assert(!v.hasNaNs());
			x -= v.x;
			y -= v.y;
			z -= v.z;
			return *this;
		}

		template <typename U>
		__host__ __device__ Point3<T> operator*(U f) const {
			assert(!isNaN(f));
			return Point3<T>(f * x, f * y, f * z);
		}

		template <typename U>
		__host__ __device__ Point3<T> operator/(U f) const {
			assert(!isNaN(f));
			float inv = (float)1 / f;
			return Point3<T>(x * inv, y * inv, z * inv);
		}

		template <typename U>
		__host__ __device__ Point3<T>& operator*=(U f) {
			assert(!isNaN(f));
			x *= f;
			y *= f;
			z *= f;
			return *this;
		}

		template <typename U>
		__host__ __device__ Point3<T>& operator/=(U f) {
			assert(!isNaN(f));
			float inv = (float)1 / f;
			x *= inv;
			y *= inv;
			z *= inv;
			return *this;
		}

		__host__ __device__ Point3<T> operator-() const { return Point3<T>(-x, -y, -z); }

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

		__host__ __device__ bool operator==(const Point3<T>& p) const { return x == p.x && y == p.y && z == p.z; }
		__host__ __device__ bool operator!=(const Point3<T>& p) const { return x != p.x || y != p.y || z != p.z; }

		__host__ __device__ void print() const { printf("Point3 -> x: %f y: %f z: %f\n", (float)x, (float)y, (float)z); }
		__host__ __device__ bool hasNaNs() const { return isNaN(x) || isNaN(y) || isNaN(z); }
	};

	// POINT TYPES
	typedef Point2<float> Point2f;
	typedef Point3<float> Point3f;
	typedef Point2<EFloat> Point2e;
	typedef Point3<EFloat> Point3e;
	typedef Point2<int> Point2i;
	typedef Point3<int> Point3i;
}

#endif