#ifndef __interaction_cuh__
#define __interaction_cuh__

#include "Geometry.cuh"
#include "../rendering/Spectrum.cuh"
#include "../materials/TransportMode.cuh"

namespace lts {

	// Forward declaration
	class Triangle;
	class Primitive;
	class BSDF;


	// add a Intersection struct that replaces the surface interaction. use it in visibility tester and normal stuff.
	// would include p, pError, wo and n

	struct Interaction {

		// Interaction definition
		Point3f p; // 3D intersection point
		Vector3f pError; // Error of 3D intersection point
		Vector3f wo; // Incomming ray direction
		Normal3f n; // Geometric normal

		__device__ Interaction() {}

		__device__ Interaction(const Point3f& p, const Normal3f& n, const Vector3f& pError,
			const Vector3f& wo) : p(p), n(n), pError(pError), wo(wo) {}

		__device__ Interaction(const Point3f& p,
			const Vector3f& wo) : p(p), wo(wo) {}

		__device__ Interaction(const Point3f& p) : p(p) {}

		__device__ Ray spawnRay(const Vector3f& d) const {
			Point3f origin = offsetRayOrigin(p, pError, n, d);
			return Ray(origin, d);
		}

		__device__ Ray spawnRayTo(const Point3f& p2) const {
			Point3f origin = offsetRayOrigin(p, pError, n, p2 - p);
			return Ray(origin, p2 - p, 1 - SHADOW_EPSILON);
		}

		__device__ Ray spawnRayTo(const Interaction& it) const {
			Point3f origin = offsetRayOrigin(p, pError, n, it.p - p);
			Point3f target = offsetRayOrigin(it.p, it.pError, it.n, origin - it.p);
			return Ray(origin, target - origin, 1 - SHADOW_EPSILON);
		}
	};

	struct SurfaceInteraction {

		Interaction it;

		const Triangle* tri = nullptr; // Intersected geometric shape
		const Primitive* primitive = nullptr; // Intersected primitive in the scene
		BSDF* bsdf = nullptr; // Material property at this point

		// Intersected surface data
		Point2f uv; // 2D intersection point
		Vector3f dpdu, dpdv; // 2D Partial derivatives at 3D intersection point
		Normal3f dndu, dndv; // 2D Partial derivatives for 3D normal
		struct {
			Normal3f n; // Shading normal 
			Vector3f dpdu, dpdv; // Shading partial derivatives at 3D intersection point
			Normal3f dndu, dndv; // Shading partial derivatives for 3D normal
		} shading;

		mutable Vector3f dpdx, dpdy; // 3D Partial derivatives at 3D intersection point
		mutable float dudx = 0, dvdx = 0, dudy = 0, dvdy = 0; // 3D partial derivatives at 2D intersection point

		__device__ SurfaceInteraction() {}

		__device__ SurfaceInteraction(const Point3f& p, const Normal3f& n, const Vector3f& pError,
			const Vector3f& wo) : it(p, n, pError, wo) {}

		__device__ SurfaceInteraction(const Point3f& p,
			const Vector3f& wo) : it(p, wo) {}

		__device__ SurfaceInteraction(const Point3f& p) : it(p) {}

		__device__ SurfaceInteraction(const Point3f& p, const Vector3f& pError, const Point2f& uv,
			const Vector3f& wo, const Vector3f& dpdu, const Vector3f& dpdv, const Normal3f& dndu, const Normal3f& dndv,
			const Triangle* tri) : it(p, Normal3f(normalize(cross(dpdu, dpdv))), pError, wo), uv(uv),
			dpdu(dpdu), dpdv(dpdv), dndu(dndu), dndv(dndv), tri(tri) {
			shading.n = it.n;
			shading.dpdu = dpdu;
			shading.dpdv = dpdv;
			shading.dndu = dndu;
			shading.dndv = dndv;
		}

		__device__ void clean();

		__device__ ~SurfaceInteraction() { clean(); }

		__device__ void setShaddingGeometry(const Vector3f& dpdus, const Vector3f dpdvs,
			const Normal3f dndus, const Normal3f& dndvs, bool orientationIsAuthoritative) {

			shading.n = normalize((Normal3f)cross(dpdus, dpdvs));

			if (orientationIsAuthoritative) {
				it.n = faceTowards(it.n, shading.n);
			}
			else {
				shading.n = faceTowards(shading.n, it.n);
			}

			shading.dpdu = dpdus;
			shading.dpdv = dpdvs;
			shading.dndu = dndus;
			shading.dndv = dndvs;
		}

		__device__ void computeDifferentials(const Ray& r) const {

			if (r.hasDifferentials) {

				float d = -dot(it.n, Vector3f(it.p.x, it.p.y, it.p.z));
				float tx = (-dot(it.n, Vector3f(r.rxOrigin)) - d) /
					dot(it.n, r.rxDirection);
				Point3f px = r.rxOrigin + r.rxDirection * tx;

				float ty = (-dot(it.n, Vector3f(r.ryOrigin)) - d) /
					dot(it.n, r.ryDirection);
				Point3f py = r.ryOrigin + r.ryDirection * ty;

				dpdx = px - it.p;
				dpdy = py - it.p;

				int dim[2];

				if (fabsf(it.n.x) > fabsf(it.n.y) && fabsf(it.n.x) > fabsf(it.n.z)) {
					dim[0] = 1; dim[1] = 2;
				}
				else if (fabsf(it.n.y) > fabsf(it.n.z)) {
					dim[0] = 0; dim[1] = 2;
				}
				else {
					dim[0] = 0; dim[1] = 1;
				}

				float A[2][2] = { {dpdu[dim[0]], dpdv[dim[0]]},
					{dpdu[dim[1]], dpdv[dim[1]]} };
				float BX[2] = { px[dim[0]] - it.p[dim[0]], px[dim[1]] - it.p[dim[1]] };
				float BY[2] = { py[dim[0]] - it.p[dim[0]], py[dim[1]] - it.p[dim[1]] };

				if (!solveLinearSystem2x2(A, BX, &dudx, &dvdx)) dudx = dvdx = 0;
				if (!solveLinearSystem2x2(A, BY, &dudy, &dvdy)) dudy = dvdy = 0;

			}
			else {
				dudx = dvdx = 0;
				dudy = dvdy = 0;
				dpdx = dpdy = Vector3f(0, 0, 0);
			}
		}

		__device__ Spectrum Le(const Vector3f& w) const;

		//add arena
		__device__ void computeScatteringFunctions(const Ray& r, TransportMode mode = TransportMode::Radiance);
	};
}

#endif