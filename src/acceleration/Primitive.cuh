#ifndef __primitive_cuh__
#define __primitive_cuh__

#include "../shapes/Triangle.cuh"
#include "../lights/Light.cuh"
#include "../materials/Material.cuh"

namespace lts {

	// Represents one triangle in the scene
	class Primitive {

	public:

		Triangle* tri;
		Material* mat = nullptr;
		AreaLight* light = nullptr;
		// const Transform PTW;

		__device__ Primitive(Triangle* t, AreaLight* l) {
			tri = t;
			light = l;
		}

		__device__ Bounds3f worldBound() const { return tri->worldBound(); }

		__device__ bool simpleIntersect(const Ray& r) const {
			return tri->simpleIntersect(r);
		}

		__device__ bool intersect(const Ray& r, SurfaceInteraction* si) const {

			float tHit;

			if (!tri->intersect(r, &tHit, si)) return false;

			r.tMax = tHit;
			si->primitive = this;

			assert(dot(si->it.n, si->shading.n) >= 0.0f);

			return true;
		}

		__device__ void setMaterial(Material* m) {
			mat = m;
		}

		__device__ AreaLight* getAreaLight() const { return light; }

		// add arena
		__device__ void computeScatteringFunctions(SurfaceInteraction* si, TransportMode mode) const {
			if (mat) {
				mat->computeScatteringFunctions(si, mode);
			}
		}
	};
}

#endif