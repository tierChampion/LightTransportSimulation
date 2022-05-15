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

	};

}

#endif