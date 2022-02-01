#ifndef __bound_cuh__
#define __bound_cuh__

#include "Ray.cuh"

namespace lts {

	template <typename T> class Bounds2 {

	public:

		Point2<T> pMin;
		Point2<T> pMax;

		__host__ __device__ Bounds2() {
			T minNum = -INFINITY;
			T maxNum = INFINITY;
			pMin = Point2<T>(maxNum, maxNum);
			pMax = Point2<T>(minNum, minNum);
		}
		__host__ __device__ Bounds2(const Point2<T>& p1, const Point2<T>& p2) {
			pMin = Point2<T>(fminf(p1.x, p2.x), fminf(p1.y, p2.y));
			pMax = Point2<T>(fmaxf(p1.x, p2.x), fmaxf(p1.y, p2.y));
		}

		__host__ __device__ explicit Bounds2(const Point2<T>& p) : pMin(p), pMax(p) {}
		template <typename U>
		__host__ __device__ explicit operator Bounds2<U>() const {
			return Bounds2<U>((Point2<U>)pMin, (Point2<U>)pMax);
		}

		__host__ __device__ Vector2<T> diagonal() const { return pMax - pMin; }

		__host__ __device__ T area() const {
			Vector2<T> d = diagonal();
			return d.x * d.y;
		}

		/**
		  Returns the longest dimension of the bound. (x = 0, y = 1)
		*/
		__host__ __device__ int maximumExtent() const {
			Vector2<T> d = diagonal();
			return (d.x <= d.y) ? 1 : 0;
		}

		__host__ __device__ inline const Point2<T>& operator[](int i) const {
			return (i == 0) ? pMin : pMax;
		}
		__host__ __device__ inline const Point2<T>& operator[](int i) {
			return (i == 0) ? pMin : pMax;
		}

		__host__ __device__ Point2<T> linearInterpolation(const Point2f& t) const {
			return Point2<T>(lts::linearInterpolation(t.x, pMin.x, pMax.x),
				lts::linearInterpolation(t.y, pMin.y, pMax.y));
		}

		/*
		  Returns the continuous position between the min corner (0, 0) and the max corner (1, 1)
		*/
		__host__ __device__ Vector2<T> offset(const Point2<T>& p) const {
			Vector2<T> o = p - pMin;
			if (pMax.x > pMin.x) o.x /= pMax.x - pMin.x;
			if (pMax.y > pMin.y) o.y /= pMax.y - pMin.y;
			return o;
		}

		__host__ __device__ void boundingCircle(Point2<T>* c, float* rad) const {
			*c = (pMin + pMax) / 2;
			*rad = inside(*c, *this) ? distance(*c, pMax) : 0;
		}

		__host__ __device__ bool operator==(const Bounds2<T>& b) const {
			return b.pMin == pMin && b.pMax == pMax;
		}
		__host__ __device__ bool operator!=(const Bounds2<T>& b) const {
			return b.pMin != pMin || b.pMax != pMax;
		}

		__host__ __device__ void print() const {
			printf("Bounds2: \n");
			printf("pMin = "); pMin.print();
			printf("pMax = "); pMax.print();
		}
	};
	// Bounds2 types
	typedef Bounds2<float> Bounds2f;
	typedef Bounds2<int> Bounds2i;

	///
	/// 3D BOUNDS
	/// 
	template <typename T> class Bounds3 {

	public:

		Point3<T> pMin;
		Point3<T> pMax;

		__host__ __device__ Bounds3() {
			T minNum = -INFINITY;
			T maxNum = INFINITY;
			pMin = Point3<T>(maxNum, maxNum, maxNum);
			pMax = Point3<T>(minNum, minNum, minNum);
		}
		__host__ __device__ Bounds3(const Point3<T>& p1, const Point3<T>& p2) {
			pMin = Point3<T>(fminf(p1.x, p2.x), fminf(p1.y, p2.y), fminf(p1.z, p2.z));
			pMax = Point3<T>(fmaxf(p1.x, p2.x), fmaxf(p1.y, p2.y), fmaxf(p1.z, p2.z));
		}

		__host__ __device__ explicit Bounds3(const Point3<T>& p) : pMin(p), pMax(p) {}
		template <typename U>
		__host__ __device__ explicit operator Bounds3<U>() const {
			return Bounds3<U>((Point3<U>)pMin, (Point3<U>)pMax);
		}

		__device__ inline bool simpleIntersect(const Ray& ray, const Vector3f& invDir,
			const int dirIsNeg[3]) const;

		/*
		  Returns the given corner of the bound (LLF = 0, LRF = 1, ULF = 2, URF = 3,
		  LLB = 4, LRB = 5, ULB = 6, URB = 7)
		*/
		__host__ __device__ Point3<T> corner(int corner) const {
			assert(corner >= 0 && corner < 8);
			return Point3<T>((*this)[(corner & 1)].x,
				(*this)[(corner & 2) ? 1 : 0].y,
				(*this)[(corner & 4) ? 1 : 0].z);
		}

		__host__ __device__ Vector3<T> diagonal() const { return pMax - pMin; }

		__host__ __device__ T volume() const {
			Vector3<T> d = diagonal();
			return d.x * d.y * d.z;
		}

		__host__ __device__ T surfaceArea() const {
			Vector3<T> d = diagonal();
			return 2 * (d.x * d.y + d.x * d.z + d.y * d.z);
		}

		/**
		  Returns the longest dimension of the bound. (x = 0, y = 1, z = 2)
		*/
		__host__ __device__ int maximumExtent() const {
			Vector3<T> d = diagonal();
			return (d.y > d.x && d.y > d.z) * 1 +
				(d.z > d.x && d.z > d.y) * 2;
		}

		__host__ __device__ inline const Point3<T>& operator[](int i) const {
			return (i == 0) ? pMin : pMax;
		}
		__host__ __device__ inline const Point3<T>& operator[](int i) {
			return (i == 0) ? pMin : pMax;
		}

		__host__ __device__ Point3<T> linearInterpolation(const Point3f& t) const {
			return Point3<T>(lts::linearInterpolation(t.x, pMin.x, pMax.x),
				lts::linearInterpolation(t.y, pMin.y, pMax.y),
				lts::linearInterpolation(t.z, pMin.z, pMax.z));
		}

		/*
		  Returns the continuous position between the min corner (0, 0, 0) and the max corner (1, 1, 1)
		*/
		__host__ __device__ Vector3<T> offset(const Point3<T>& p) const {
			Vector3<T> o = p - pMin;
			if (pMax.x > pMin.x) o.x /= pMax.x - pMin.x;
			if (pMax.y > pMin.y) o.y /= pMax.y - pMin.y;
			if (pMax.z > pMin.z) o.z /= pMax.z - pMin.z;
			return o;
		}

		__host__ __device__ void boundingSphere(Point3<T>* c, float* rad) const {
			*c = (pMin + pMax) / 2;
			*rad = inside(*c, *this) ? distance(*c, pMax) : 0;
		}

		__host__ __device__ bool operator==(const Bounds3<T>& b) const {
			return b.pMin == pMin && b.pMax == pMax;
		}
		__host__ __device__ bool operator!=(const Bounds3<T>& b) const {
			return b.pMin != pMin || b.pMax != pMax;
		}

		__host__ __device__ void print() const {
			printf("Bounds3: \n");
			printf("pMin = "); pMin.print();
			printf("pMax = "); pMax.print();
		}
	};
	// Bounds3 types
	typedef Bounds3<float> Bounds3f;
	typedef Bounds3<int> Bounds3i;

	template <typename T>
	__device__ inline bool Bounds3<T>::simpleIntersect(const Ray& ray, const Vector3f& invDir,
		const int dirIsNeg[3]) const {

		const Bounds3f& bounds = *this;
		// Check for ray intersection against $x$ and $y$ slabs
		float tMin = (bounds[dirIsNeg[0]].x - ray.o.x) * invDir.x;
		float tMax = (bounds[1 - dirIsNeg[0]].x - ray.o.x) * invDir.x;
		float tyMin = (bounds[dirIsNeg[1]].y - ray.o.y) * invDir.y;
		float tyMax = (bounds[1 - dirIsNeg[1]].y - ray.o.y) * invDir.y;

		// Update _tMax_ and _tyMax_ to ensure robust bounds intersection
		tMax *= 1 + 2 * gamma(3);
		tyMax *= 1 + 2 * gamma(3);
		if (tMin > tyMax || tyMin > tMax) return false;
		if (tyMin > tMin) tMin = tyMin;
		if (tyMax < tMax) tMax = tyMax;

		// Check for ray intersection against $z$ slab
		float tzMin = (bounds[dirIsNeg[2]].z - ray.o.z) * invDir.z;
		float tzMax = (bounds[1 - dirIsNeg[2]].z - ray.o.z) * invDir.z;

		// Update _tzMax_ to ensure robust bounds intersection
		tzMax *= 1 + 2 * gamma(3);
		if (tMin > tzMax || tzMin > tMax) return false;
		if (tzMin > tMin) tMin = tzMin;
		if (tzMax < tMax) tMax = tzMax;
		return (tMin < ray.tMax) && (tMax > 0);
	}
}

#endif