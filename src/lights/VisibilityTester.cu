#ifndef __visibility_tester_cu__
#define __visibility_tester_cu__

#include "VisibilityTester.cuh"
#include "../acceleration/Scene.cuh"

namespace lts {

	__device__ bool VisibilityTester::unoccluded(Scene scene) {

		return !scene.simpleIntersect(p0.spawnRayTo(p1));
	}

}

#endif