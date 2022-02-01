#ifndef __geometry_cuh__
#define __geometry_cuh__

#include "Bound.cuh"
#include "Normal.cuh"
#include "Ray.cuh"

namespace lts {

	/**
	* Quake III algorithm for calculating the inverse square root of a number.
	* @param f - Value to inverse under the square root
	* @return 1 / sqrt(f)
	*/
	template <typename T>
	__host__ __device__ inline float quakeIIIFastInvSqrt(T f) {
		long i;
		float x2, y;
		const float threeHalfs = 1.5f;

		x2 = f * 0.5f;
		y = f;
		i = *(long*)&y;
		i = 0x5f3759df - (i >> 1);
		y = *(float*)&i;
		y = y * (threeHalfs - (x2 * y * y));
		y = y * (threeHalfs - (x2 * y * y));
		return y;
	}

	/**
	* Dot product of two 2D vectors.
	* @param v1 - 2D vector
	* @param v2 - 2D vector
	*/
	template <typename T>
	__host__ __device__ inline T dot(const Vector2<T>& v1, const Vector2<T>& v2) {
		return v1.x * v2.x + v1.y * v2.y;
	}

	/**
	* Dot product of two 3D vectors.
	* @param v1 - 3D vector
	* @param v2 - 3D vector
	*/
	template <typename T>
	__host__ __device__ inline T dot(const Vector3<T>& v1, const Vector3<T>& v2) {
		return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
	}

	/**
	* Absolute value of the dot product of two 2D vectors.
	* @param v1 - 2D vector
	* @param v2 - 2D vector
	*/
	template <typename T>
	__host__ __device__ inline T absDot(const Vector2<T>& v1, const Vector2<T>& v2) {
		return fabsf(dot(v1, v2));
	}

	/**
	* Absolute value of the dot product of two 3D vectors.
	* @param v1 - 3D vector
	* @param v2 - 3D vector
	*/
	template <typename T>
	__host__ __device__ inline T absDot(const Vector3<T>& v1, const Vector3<T>& v2) {
		return fabsf(dot(v1, v2));
	}

	/**
	* Cross product of two 3D vectors.
	* @param v1 - 3D vector
	* @param v2 - 3D vector
	*/
	template <typename T>
	__host__ __device__ inline Vector3<T> cross(const Vector3<T>& v1, const Vector3<T>& v2) {
		double v1x = v1.x, v1y = v1.y, v1z = v1.z;
		double v2x = v2.x, v2y = v2.y, v2z = v2.z;
		return Vector3<T>((v1y * v2z) - (v1z * v2y),
			(v1z * v2x) - (v1x * v2z),
			(v1x * v2y) - (v1y * v2x));
	}

	/**
	* Normalize a given 2D vector
	* @param v - 2D vector to normalize
	*/
	template <typename T>
	__host__ __device__ inline Vector2<T> normalize(const Vector2<T>& v) {
		return v * quakeIIIFastInvSqrt(v.lengthSquared());
	}

	/**
	* Normalize a given 3D vector
	* @param v - 3D vector to normalize
	*/
	template <typename T>
	__host__ __device__ inline Vector3<T> normalize(const Vector3<T>& v) {
		return v * quakeIIIFastInvSqrt(v.lengthSquared());
	}

	/**
	* Minimum value in the components of a 2D vector
	* @param v - 2D vector
	*/
	template<typename T>
	__host__ __device__ inline T minComponent(const Vector2<T>& v) {
		return fminf(v.x, v.y);
	}

	/**
	* Minimum value in the components of a 3D vector
	* @param v - 3D vector
	*/
	template<typename T>
	__host__ __device__ inline T minComponent(const Vector3<T>& v) {
		return fminf(v.x, fminf(v.y, v.z));
	}

	/**
	* Maximum value in the components of a 2D vector
	* @param v - 2D vector
	*/
	template<typename T>
	__host__ __device__ inline T maxComponent(const Vector2<T>& v) {
		return fmaxf(v.x, v.y);
	}

	/**
	* Maximum value in the components of a 3D vector
	* @param v - 3D vector
	*/
	template<typename T>
	__host__ __device__ inline T maxComponent(const Vector3<T>& v) {
		return fmaxf(v.x, fmaxf(v.y, v.z));
	}

	/**
	* Maximum dimension in the components of a 2D vector
	* @param v - 2D vector
	*/
	template<typename T>
	__host__ __device__ inline int maxDimension(const Vector2<T>& v) {
		return (v.x > v.y) ? 0 : 1;
	}

	/**
	* Maximum dimension in the components of a 3D vector
	* @param v - 3D vector
	*/
	template<typename T>
	__host__ __device__ inline int maxDimension(const Vector3<T>& v) {
		return (v.x > v.y) ? ((v.x > v.z) ? 0 : 2) :
			((v.y > v.z) ? 1 : 2);
	}

	/**
	* Minimum components of two 2D vectors
	* @param v1 - 2D vector
	* @param v2 - 2D vector
	*/
	template<typename T>
	__host__ __device__ inline Vector2<T> min(const Vector2<T>& v1, const Vector2<T>& v2) {
		return Vector2<T>(fminf(v1.x, v2.x),
			fminf(v1.y, v2.y));
	}

	/**
	* Minimum components of two 3D vectors
	* @param v1 - 3D vector
	* @param v2 - 3D vector
	*/
	template<typename T>
	__host__ __device__ inline Vector3<T> min(const Vector3<T>& v1, const Vector3<T>& v2) {
		return Vector3<T>(fminf(v1.x, v2.x),
			fminf(v1.y, v2.y),
			fminf(v1.z, v2.z));
	}

	/**
	* Maximum components of two 2D vectors
	* @param v1 - 2D vector
	* @param v2 - 2D vector
	*/
	template<typename T>
	__host__ __device__ inline Vector2<T> max(const Vector2<T>& v1, const Vector2<T>& v2) {
		return Vector2<T>(fmaxf(v1.x, v2.x),
			fmaxf(v1.y, v2.y));
	}

	/**
	* Maximum components of two 3D vectors
	* @param v1 - 3D vector
	* @param v2 - 3D vector
	*/
	template<typename T>
	__host__ __device__ inline Vector3<T> max(const Vector3<T>& v1, const Vector3<T>& v2) {
		return Vector3<T>(fmaxf(v1.x, v2.x),
			fmaxf(v1.y, v2.y),
			fmaxf(v1.z, v2.z));
	}

	/**
	* Permutes the components of a 2D vector
	* @param v - 2D vector
	* @param x - Index of the new x component
	* @param y - Index of the new y component
	*/
	template<typename T>
	__host__ __device__ inline Vector2<T> permute(const Vector2<T>& v, int x, int y) {
		return Vector2<T>(v[x], v[y]);
	}

	/**
	* Permutes the components of a 3D vector
	* @param v - 3D vector
	* @param x - Index of the new x component
	* @param y - Index of the new y component
	* @param z - Index of the new z component
	*/
	template<typename T>
	__host__ __device__ inline Vector3<T> permute(const Vector3<T>& v, int x, int y, int z) {
		return Vector3<T>(v[x], v[y], v[z]);
	}

	/**
	* Absolute value of a 2D vector
	* @param v - 2D vector
	*/
	template<typename T>
	__host__ __device__ inline Vector2<T> abs(const Vector2<T>& v) {
		return Vector2<T>(fabsf(v.x), fabsf(v.y));
	}

	/**
	* Absolute value of a 3D vector
	* @param v - 3D vector
	*/
	template<typename T>
	__host__ __device__ inline Vector3<T> abs(const Vector3<T>& v) {
		return Vector3<T>(fabsf(v.x), fabsf(v.y), fabsf(v.z));
	}

	/**
	* Construct an arbitrary XYZ coordinate system from a given v1 vector.
	* @param v1 - 3D vector to build the coordinate system upon. Must be normalized
	* @param v2 and v3 - New 3D vector axis
	*/
	template <typename T>
	__host__ __device__ inline void coordinateSystem(const Vector3<T>& v1, Vector3<T>* v2, Vector3<T>* v3) {

		if (fabsf(v1.x) > fabsf(v1.y))
			*v2 = Vector3<T>(-v1.z, 0, v1.x) * quakeIIIFastInvSqrt(v1.x * v1.x + v1.z * v1.z);
		else
			*v2 = Vector3<T>(0, v1.z, -v1.y) * quakeIIIFastInvSqrt(v1.y * v1.y + v1.z * v1.z);

		*v3 = cross(v1, *v2);
	}

	///
	/// POINT STATIC METHODS
	///

	/**
	* Distance between two 2D points.
	* @param p1 - 2D point
	* @param p2 - 2D point
	*/
	template <typename T>
	__host__ __device__ inline float distance(const Point2<T>& p1, const Point2<T>& p2) {
		return (p1 - p2).length();
	}

	/**
	* Distance between two 3D points.
	* @param p1 - 3D point
	* @param p2 - 3D point
	*/
	template <typename T>
	__host__ __device__ inline float distance(const Point3<T>& p1, const Point3<T>& p2) {
		return (p1 - p2).length();
	}

	/**
	* Squared Distance between 2D two points.
	* @param p1 - 2D point
	* @param p2 - 2D point
	*/
	template <typename T>
	__host__ __device__ inline float distanceSquared(const Point2<T>& p1, const Point2<T>& p2) {
		return (p1 - p2).lengthSquared();
	}

	/**
	* Squared Distance between 3D two points.
	* @param p1 - 3D point
	* @param p2 - 3D point
	*/
	template <typename T>
	__host__ __device__ inline float distanceSquared(const Point3<T>& p1, const Point3<T>& p2) {
		return (p1 - p2).lengthSquared();
	}

	/**
	* Linear interpolation between two 2D points.
	* @param t - Time of interolation
	* @param p0 - Initial 2D point
	* @param pf - Final 2D point
	*/
	template <typename T>
	__host__ __device__ inline Point2<T> linearInterpolation(float t, const Point2<T>& p0, const Point2<T>& pf) {
		return (1 - t) * p0 + t * pf;
	}

	/**
	* Linear interpolation between two 3D points.
	* @param t - Time of interolation
	* @param p0 - Initial 3D point
	* @param pf - Final 3D point
	*/
	template <typename T>
	__host__ __device__ inline Point3<T> linearInterpolation(float t, const Point3<T>& p0, const Point3<T>& pf) {
		return (1 - t) * p0 + t * pf;
	}

	/**
	* Minimum components of two 2D points
	* @param p1 - 2D point
	* @param p2 - 2D point
	*/
	template<typename T>
	__host__ __device__ inline Point2<T> min(const Point2<T>& p1, const Point2<T>& p2) {
		return Point2<T>(fminf(p1.x, p2.x),
			fminf(p1.y, p2.y));
	}

	/**
	* Minimum components of two 3D points
	* @param p1 - 3D point
	* @param p2 - 3D point
	*/
	template<typename T>
	__host__ __device__ inline Point3<T> min(const Point3<T>& p1, const Point3<T>& p2) {
		return Point3<T>(fminf(p1.x, p2.x),
			fminf(p1.y, p2.y),
			fminf(p1.z, p2.z));
	}

	/**
	* Maximum components of two 2D points
	* @param p1 - 2D point
	* @param p2 - 2D point
	*/
	template<typename T>
	__host__ __device__ inline Point2<T> max(const Point2<T>& p1, const Point2<T>& p2) {
		return Point2<T>(fmaxf(p1.x, p2.x),
			fmaxf(p1.y, p2.y));
	}

	/**
	* Maximum components of two 3D points
	* @param p1 - 3D point
	* @param p2 - 3D point
	*/
	template<typename T>
	__host__ __device__ inline Point3<T> max(const Point3<T>& p1, const Point3<T>& p2) {
		return Point3<T>(fmaxf(p1.x, p2.x),
			fmaxf(p1.y, p2.y),
			fmaxf(p1.z, p2.z));
	}

	/**
	* Floor of a 2D point
	* @param p - 2D point
	*/
	template <typename T>
	__host__ __device__ inline Point2<T> floor(const Point2<T>& p) {
		return Point2<T>(floorf(p.x),
			floorf(p.y));
	}

	/**
	* Floor of a 3D point
	* @param p - 3D point
	*/
	template <typename T>
	__host__ __device__ inline Point3<T> floor(const Point3<T>& p) {
		return Point3<T>(floorf(p.x),
			floorf(p.y),
			floorf(p.z));
	}

	/**
	* Ceil of a 2D point
	* @param p - 2D point
	*/
	template <typename T>
	__host__ __device__ inline Point2<T> ceil(const Point2<T>& p) {
		return Point2<T>(ceilf(p.x),
			ceilf(p.y));
	}

	/**
	* Ceil of a 3D point
	* @param p - 3D point
	*/
	template <typename T>
	__host__ __device__ inline Point3<T> ceil(const Point3<T>& p) {
		return Point3<T>(ceilf(p.x),
			ceilf(p.y),
			ceilf(p.z));
	}

	/**
	* Absolute value of a 2D point
	* @param p - 2D point
	*/
	template <typename T>
	__host__ __device__ inline Point2<T> abs(const Point2<T>& p) {
		return Point2<T>(absf(p.x),
			absf(p.y));
	}

	/**
	* Absolute value of a 3D point
	* @param p - 3D point
	*/
	template <typename T>
	__host__ __device__ inline Point3<T> abs(const Point3<T>& p) {
		return Point3<T>(fabsf(p.x),
			fabsf(p.y),
			fabsf(p.z));
	}

	/**
	* Permutes the components of a 2D point
	* @param p - 2D point
	* @param x - Index of the new x component
	* @param y - Index of the new y component
	*/
	template<typename T>
	__host__ __device__ inline Point2<T> permute(const Point2<T>& p, int x, int y) {
		return Point2<T>(p[x], p[y]);
	}

	/**
	* Permutes the components of a 3D point
	* @param p - 3D point
	* @param x - Index of the new x component
	* @param y - Index of the new y component
	* @param z - Index of the new z component
	*/
	template<typename T>
	__host__ __device__ inline Point3<T> permute(const Point3<T>& p, int x, int y, int z) {
		return Point3<T>(p[x], p[y], p[z]);
	}

	///
	/// NORMAL STATIC METHODS
	///

	/**
	* Normalize a given 3D normal
	* @param n - 3D normal to normalize
	*/
	template<typename T>
	__host__ __device__ inline Normal3<T> normalize(const Normal3<T>& n) {
		return n * quakeIIIFastInvSqrt(n.lengthSquared());
	}

	/**
	* Dot product of a 3D normal and a 3D vector.
	* @param n - 3D normal
	* @param v - 3D vector
	*/
	template<typename T>
	__host__ __device__ inline T dot(const Normal3<T>& n, const Vector3<T>& v) {
		return n.x * v.x + n.y * v.y + n.z * v.z;
	}

	/**
	* Dot product of a 3D vector and a 3D normal.
	* @param v - 3D vector
	* @param n - 3D normal
	*/
	template<typename T>
	__host__ __device__ inline T dot(const Vector3<T>& v, const Normal3<T>& n) {
		return dot(n, v);
	}

	/**
	* Dot product of two 3D normals.
	* @param n1 - 3D normal
	* @param n2 - 3D normal
	*/
	template<typename T>
	__host__ __device__ inline T dot(const Normal3<T>& n1, const Normal3<T>& n2) {
		return n1.x * n2.x + n1.y * n2.y + n1.z * n2.z;
	}

	/**
	* Absolute value of the dot product of a 3D normal and a 3D vector.
	* @param n - 3D vector
	* @param v - 3D vector
	*/
	template<typename T>
	__host__ __device__ inline T absDot(const Normal3<T>& n, const Vector3<T>& v) {
		return fabsf(dot(n, v));
	}

	/**
	* Absolute value of the dot product of a 3D vector and a 3D normal.
	* @param v - 3D vector
	* @param n - 3D normal
	*/
	template<typename T>
	__host__ __device__ inline T absDot(const Vector3<T>& v, const Normal3<T>& n) {
		return fabsf(dot(n, v));
	}

	/**
	* Absolute value of the dot product of two 3D normals.
	* @param n1 - 3D normal
	* @param n2 - 3D normal
	*/
	template<typename T>
	__host__ __device__ inline T absDot(const Normal3<T>& n1, const Normal3<T>& n2) {
		return fabsf(dot(n1, n2));
	}

	/**
	* Gives a 3D normal facing towards a 3D vector
	* @param n - 3D normal
	* @param v - 3D vector
	*/
	template<typename T>
	__host__ __device__ inline Normal3<T> faceTowards(const Normal3<T>& n, const Vector3<T>& v) {
		return (dot(n, v) < 0.0f) ? -n : n;
	}

	/**
	* Gives a 3D vector facing towards a 3D normal
	* @param v - 3D vector
	* @param n - 3D normal
	*/
	template<typename T>
	__host__ __device__ inline Vector3<T> faceTowards(const Vector3<T>& v, const Normal3<T>& n) {
		return (dot(v, n) < 0.0f) ? -v : v;
	}

	/**
	* Gives a 3D normal facing towards another 3D normal
	* @param n1 - 3D normal
	* @param n2 - 3D normal
	*/
	template<typename T>
	__host__ __device__ inline Normal3<T> faceTowards(const Normal3<T>& n1, const Normal3<T>& n2) {
		return (dot(n1, n2) < 0.0f) ? -n1 : n1;
	}

	/**
	* Gives a 3D vector facing towards another 3D vector
	* @param n - 3D normal
	* @param v - 3D vector
	*/
	template<typename T>
	__host__ __device__ inline Vector3<T> faceTowards(const Vector3<T>& v1, const Vector3<T>& v2) {
		return (dot(v1, v2) < 0.0f) ? -v1 : v1;
	}

	/**
	* Absolute value of a 3D normal
	* @param n - 3D normal
	*/
	template<typename T>
	__host__ __device__ inline Normal3<T> abs(const Normal3<T>& n) {
		return Normal3<T>(fabsf(n.x), fabsf(n.y), fabsf(n.z));
	}

	///
	/// BOUNDS STATIC METHODS
	///

	/**
	* Gives the 2D bounding box enclosing another 2D bounding box and a 2D point
	* @param b - 2D bounding box
	* @param p - 2D point
	*/
	template <typename T>
	__host__ __device__ inline Bounds2<T> bUnion(const Bounds2<T>& b, const Point2<T>& p) {
		return Bounds2<T>(Point2<T>(fminf(b.pMin.x, p.x),
			fminf(b.pMin.y, p.y)),
			Point2<T>(fmaxf(b.pMax.x, p.x),
				fmaxf(b.pMax.y, p.y)));
	}

	/**
	* Gives the 3D bounding box enclosing another 3D bounding box and a 3D point
	* @param b - 3D bounding box
	* @param p - 3D point
	*/
	template <typename T>
	__host__ __device__ inline Bounds3<T> bUnion(const Bounds3<T>& b, const Point3<T>& p) {
		return Bounds3<T>(Point3<T>(fminf(b.pMin.x, p.x),
			fminf(b.pMin.y, p.y),
			fminf(b.pMin.z, p.z)),
			Point3<T>(fmaxf(b.pMax.x, p.x),
				fmaxf(b.pMax.y, p.y),
				fmaxf(b.pMax.z, p.z)));
	}

	/**
	* Gives the 2D bounding box enclosing two other 2D bounding boxes
	* @param b1 - 2D bounding box
	* @param b2 - 2D bounding box
	*/
	template <typename T>
	__host__ __device__ inline Bounds2<T> bUnion(const Bounds2<T>& b1, const Bounds2<T>& b2) {
		return Bounds2<T>(Point2<T>(fminf(b1.pMin.x, b2.pMin.x),
			fminf(b1.pMin.y, b2.pMin.y)),
			Point2<T>(fmaxf(b1.pMax.x, b2.pMax.x),
				fmaxf(b1.pMax.y, b2.pMax.y)));
	}

	/**
	* Gives the 3D bounding box enclosing two other 3D bounding boxes
	* @param b1 - 3D bounding box
	* @param b2 - 3D bounding box
	*/
	template <typename T>
	__host__ __device__ inline Bounds3<T> bUnion(const Bounds3<T>& b1, const Bounds3<T>& b2) {
		return Bounds3<T>(Point3<T>(fminf(b1.pMin.x, b2.pMin.x),
			fminf(b1.pMin.y, b2.pMin.y),
			fminf(b1.pMin.z, b2.pMin.z)),
			Point3<T>(fmaxf(b1.pMax.x, b2.pMax.x),
				fmaxf(b1.pMax.y, b2.pMax.y),
				fmaxf(b1.pMax.z, b2.pMax.z)));
	}

	/**
	* Gives the 3D bounding box enclosing two other 3D bounding boxes
	* @param b1 - 3D bounding box to modify
	* @param b2 - 3D bounding box
	*/
	template <typename T>
	__host__ __device__ inline void atomicUnion(Bounds3<T>* b1, const Bounds3<T>& b2) {
		fatomicMin(&b1->pMin.x, b2.pMin.x);
		fatomicMin(&b1->pMin.y, b2.pMin.y);
		fatomicMin(&b1->pMin.z, b2.pMin.z);
		fatomicMax(&b1->pMax.x, b2.pMax.x);
		fatomicMax(&b1->pMax.y, b2.pMax.y);
		fatomicMax(&b1->pMax.z, b2.pMax.z);
	}

	/**
	* Gives the intersection of two 2D bounding boxes
	* @param b1 - 2D bounding box
	* @param b2 - 2D bounding box
	*/
	template <typename T>
	__host__ __device__ inline Bounds2<T> intersection(const Bounds2<T>& b1, const Bounds2<T>& b2) {
		return Bounds2<T>(Point2<T>(fmaxf(b1.pMin.x, b2.pMin.x),
			fmaxf(b1.pMin.y, b2.pMin.y)),
			Point2<T>(fminf(b1.pMax.x, b2.pMax.x),
				fminf(b1.pMax.y, b2.pMax.y)));
	}

	/**
	* Gives the intersection of two 3D bounding boxes
	* @param b1 - 3D bounding box
	* @param b2 - 3D bounding box
	*/
	template <typename T>
	__host__ __device__ inline Bounds3<T> intersection(const Bounds3<T>& b1, const Bounds3<T>& b2) {
		return Bounds3<T>(Point3<T>(fmaxf(b1.pMin.x, b2.pMin.x),
			fmaxf(b1.pMin.y, b2.pMin.y),
			fmaxf(b1.pMin.z, b2.pMin.z)),
			Point3<T>(fminf(b1.pMax.x, b2.pMax.x),
				fminf(b1.pMax.y, b2.pMax.y),
				fminf(b1.pMax.z, b2.pMax.z)));
	}

	/**
	* Determines if two 2D bounding boxes overlap
	* @param b1 - 2D bounding box
	* @param b2 - 2D bounding box
	*/
	template <typename T>
	__host__ __device__ inline bool overlaps(const Bounds2<T>& b1, const Bounds2<T>& b2) {
		return ((b1.pMax.x >= b2.pMin.x) && (b1.pMin.x <= b2.pMax.x)) &&
			((b1.pMax.y >= b2.pMin.y) && (b1.pMin.y <= b2.pMax.y));
	}

	/**
	* Determines if two 3D bounding boxes overlap
	* @param b1 - 3D bounding box
	* @param b2 - 3D bounding box
	*/
	template <typename T>
	__host__ __device__ inline bool overlaps(const Bounds3<T>& b1, const Bounds3<T>& b2) {
		return ((b1.pMax.x >= b2.pMin.x) && (b1.pMin.x <= b2.pMax.x)) &&
			((b1.pMax.y >= b2.pMin.y) && (b1.pMin.y <= b2.pMax.y)) &&
			((b1.pMax.z >= b2.pMin.z) && (b1.pMin.z <= b2.pMax.z));
	}

	/**
	* Determines if a 2D point is inside a 2D bounding box
	* @param p - 2D point
	* @param b - 2D bounding box
	*/
	template <typename T>
	__host__ __device__ inline bool inside(const Point2<T>& p, const Bounds2<T>& b) {
		return ((p.x >= b.pMin.x) && (p.x <= b.pMax.x)) &&
			((p.y >= b.pMin.y) && (p.y <= b.pMax.y));
	}

	/**
	* Determines if a 2D point is inside a 2D bounding box
	* @param p - 2D point
	* @param b - 2D bounding box
	*/
	template <typename T>
	__host__ __device__ inline bool insideExclusive(const Point2<T>& p, const Bounds2<T>& b) {
		return ((p.x >= b.pMin.x) && (p.x < b.pMax.x)) &&
			((p.y >= b.pMin.y) && (p.y < b.pMax.y));
	}

	/**
	* Determines if a 3D point is inside a 3D bounding box
	* @param p - 3D point
	* @param b - 3D bounding box
	*/
	template <typename T>
	__host__ __device__ inline bool inside(const Point3<T>& p, const Bounds3<T>& b) {
		return ((p.x >= b.pMin.x) && (p.x <= b.pMax.x)) &&
			((p.y >= b.pMin.y) && (p.y <= b.pMax.y)) &&
			((p.z >= b.pMin.z) && (p.z <= b.pMax.z));
	}

	/**
	* Determines if a 3D point is inside a 3D bounding box
	* @param p - 3D point
	* @param b - 3D bounding box
	*/
	template <typename T>
	__host__ __device__ inline bool insideExclusive(const Point3<T>& p, const Bounds3<T>& b) {
		return ((p.x >= b.pMin.x) && (p.x < b.pMax.x)) &&
			((p.y >= b.pMin.y) && (p.y < b.pMax.y)) &&
			((p.z >= b.pMin.z) && (p.z < b.pMax.z));
	}

	/**
	* Enlarges a 2D bounding box by a given delta in all dimensions
	* @param b - 2D bounding box
	* @param halfDelta - Half displacement to the corners in all dimensions
	*/
	template <typename T, typename U>
	__host__ __device__ inline Bounds2<T> expand(const Bounds2<T>& b, U halfDelta) {
		return Bounds2<T>(b.pMin - Vector2<T>(halfDelta, halfDelta),
			b.pMax + Vector2<T>(halfDelta, halfDelta));
	}

	/**
	* Enlarges a 3D bounding box by a given delta in all dimensions
	* @param b - 3D bounding box
	* @param halfDelta - Half displacement to the corners in all dimensions
	*/
	template <typename T, typename U>
	__host__ __device__ inline Bounds3<T> expand(const Bounds3<T>& b, U halfDelta) {
		return Bounds3<T>(b.pMin - Vector3<T>(halfDelta, halfDelta, halfDelta),
			b.pMax + Vector3<T>(halfDelta, halfDelta, halfDelta));
	}

	/**
	* Offset the ray origin by a factor of the surface normal in order to avoid self-intersection.
	* @param p - 3D point, origin of the ray
	* @param pError - Error on the ray origin, minimum distance to travel to leave the error bound
	* @param n - 3D normal, surface normal
	* @param w - 3D vector, exit direction
	*/
	__device__ inline Point3f offsetRayOrigin(const Point3f& p, const Vector3f& pError,
		const Normal3f& n, const Vector3f& w) {
		float d = dot(abs(n), pError);
		Vector3f offset = Vector3f(n) * d;
		if (dot(w, n) < 0) offset = -offset;
		Point3f po = p + offset;

		for (int i = 0; i < 3; i++) {
			if (offset[i] > 0)
				po[i] = nextFloatUp(po[i]);
			else if (offset[i] < 0)
				po[i] = nextFloatDown(po[i]);
		}
		return po;
	}

	/**
	* Builds a 3D unit vector from two given angles phi and theta
	* @param sinTheta and cosTheta - The sin and the cos of the theta angle, or meridian
	* @param phi - Phi angle, or latitude
	*/
	__host__ __device__ inline Vector3f sphericalDirection(float sinTheta, float cosTheta, float phi) {
		return Vector3f(sinTheta * cosf(phi),
			sinTheta * sinf(phi),
			cosTheta);
	}

	/**
	* Builds a 3D unit vector from two given angles phi and theta and a given coordinate system
	* @param sinTheta and cosTheta - The sin and the cos of the theta angle, or meridian
	* @param phi - Phi angle, or latitude
	* @param x, y and z - The three axis of the coordinate system
	*/
	__host__ __device__ inline Vector3f sphericalDirection(float sinTheta, float cosTheta, float phi,
		const Vector3f& x, const Vector3f& y, const Vector3f& z) {
		return x * (sinTheta * cosf(phi)) +
			y * (sinTheta * sinf(phi)) +
			z * cosTheta;
	}

	/**
	* Calculte the corresponding theta angle of a 3D vector
	* @param v - 3D vector
	*/
	__host__ __device__ inline float sphericalTheta(const Vector3f& v) {
		return acosf(clamp(v.z, -1.0f, 1.0f));
	}

	/**
	* Calculate the corresponding phi angle of a 3D vector
	* @param v - 3D vector
	*/
	__host__ __device__ inline float SphericalPhi(const Vector3f& v) {
		float p = atan2f(v.y, v.x);
		return (p < 0) ? (p + 2 * M_PI) : p;
	}
}

#endif