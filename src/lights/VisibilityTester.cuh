#ifndef __visibility_cuh__
#define __visibility_cuh__

#include "../geometry/Interaction.cuh"

namespace lts {

	class Scene;

	class VisibilityTester {

	public:

		Interaction p0;
		Interaction p1;

		__device__ VisibilityTester() {}
		__device__ VisibilityTester(const Interaction& P0, const Interaction& P1) : p0(P0), p1(P1) {}

		__device__ bool unoccluded(Scene scene);

		__device__ void info() {
			printf("Testing if (%f %f %f) sees (%f %f %f).\n", p0.p.x, p0.p.y, p0.p.z,
				p1.p.x, p1.p.y, p1.p.z);
		}

	};

}

#endif