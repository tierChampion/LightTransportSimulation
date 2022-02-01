#ifndef __ray_cuh__
#define __ray_cuh__

#include "Point.cuh"

namespace lts {

	class Ray {

	public:

		Point3f o;
		Vector3f d;
		mutable float tMax;
		// Differential part of ray
		bool hasDifferentials = false;
		Point3f rxOrigin, ryOrigin; // change of ray origin per x and y shift on the film
		Vector3f rxDirection, ryDirection; // same thing but for the direction


		__device__ Ray() : tMax(INFINITY) {}
		__device__ Ray(const Point3f& o, const Vector3f& d, float tMax = INFINITY)
			: o(o), d(d), tMax(tMax) {}

		__device__ Point3f operator()(float t) const { return o + d * t; }

		__device__ void print() const {
			printf("Ray:\n");
			printf("o = ");  o.print();
			printf("d = "); d.print();
			printf("tMax = %f\n", tMax);
		}
		__device__ bool hasNaNs() const { return o.hasNaNs() || d.hasNaNs() || isnan(tMax); }

		__device__ void scaleDifferentials(float scale) {
			rxOrigin = o + (rxOrigin - o) * scale;
			ryOrigin = o + (ryOrigin - o) * scale;
			rxDirection = d + (rxDirection - d) * scale;
			ryDirection = d + (ryDirection - d) * scale;
		}
	};
}

#endif