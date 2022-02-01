#ifndef __mapping_cuh__
#define __mapping_cuh__

#include "../../geometry/Transform.cuh"

namespace lts {

	// 2d mapping
	class TextureMapping2D {

	public:

		__device__ virtual ~TextureMapping2D() {}
		__device__ virtual Point2f map(const SurfaceInteraction& si, Vector2f* dstdx, Vector2f* dstdy) const = 0;

	};

	// uv mapping, circle mapping, cylindrical mapping, planar mapping
	class UVMapping2D : public TextureMapping2D {

		const float su, sv, du, dv; // scale, offset

	public:

		__device__ UVMapping2D(float su, float sv, float du, float dv) :
			su(su), sv(sv), du(du), dv(dv) {}

		__device__ Point2f map(const SurfaceInteraction& si, Vector2f* dstdx, Vector2f* dstdy) const override {

			*dstdx = Vector2f(si.dudx * su, si.dvdx * sv);
			*dstdy = Vector2f(si.dudy * su, si.dvdy * sv);

			return Point2f(su * si.uv[0] + du,
				sv * si.uv[1] + dv);
		}
	};

	// 3d mapping
	class TextureMapping3D {

		const Transform WTT;

	public:

		__device__ TextureMapping3D(Transform worldToTexture) : WTT(worldToTexture) {}

		__device__ Point3f map(const SurfaceInteraction& si, Vector3f* dpdx, Vector3f* dpdy) const {

			*dpdx = WTT(si.dpdx);
			*dpdy = WTT(si.dpdy);

			return WTT(si.it.p);
		}
	};

}

#endif