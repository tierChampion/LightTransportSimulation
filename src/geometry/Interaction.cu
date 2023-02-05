#ifndef __interaction_cu__
#define __interaction_cu__

#include "Interaction.cuh"

#include "../acceleration/Primitive.cuh"

namespace lts {

	__device__ void SurfaceInteraction::clean() {

		if (bsdf) {
			bsdf->clean();
		}
		free(bsdf);
	}

	__device__ Spectrum SurfaceInteraction::Le(const Vector3f& w) const {
		const AreaLight* area = primitive->getAreaLight();
		return area ? area->L(this->it, w) : Spectrum(0.0f);
	}

	__device__ void SurfaceInteraction::computeScatteringFunctions(const Ray& r, TransportMode mode) {

		computeDifferentials(r);
		primitive->computeScatteringFunctions(this, mode);
	}
}

#endif