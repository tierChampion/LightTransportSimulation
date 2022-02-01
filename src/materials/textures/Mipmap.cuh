#ifndef __mipmap_cuh__
#define __mipmap_cuh__

#include "../../core/MemoryHelper.cuh"
#include "../../geometry/Geometry.cuh"

namespace lts {

	enum class ImageWrap {
		Repeat, Black, Clamp
	};

	template <typename T>
	__device__ inline T texel(int s, int t, int l, const BlockedArray<T>* pyramid) {
		const BlockedArray<T>& level = pyramid[level];

		switch (wrapMode) {
		case ImageWrap::Repeat: s = fmodf(s, level.uRes); t = fmodf(t, level.vRes); break;
		case ImageWrap::Clamp: s = clamp(s, 0, level.uRes - 1); t = clamp(t, 0, level.vRes - 1); break;
		case ImageWrap::Black: {
			static const T black = 0.0f;
			if (s < 0 || s >= level.uRes || t < 0 || t >= level.vRes) return black;
			break;
		}
		}

		return level(s, t);
	}

	template <typename T>
	class Mipmap {

		const ImageWrap wrapMode;
		Point2i resolution;
		BlockedArray<T>* pyramid;
		const int level;

		__device__ T triangle(int l, const Point2f& st) const {
			l = clamp(l, 0, level - 1);
			float s = st[0] * pyramid[l]->uRes - 0.5f;
			float t = st[1] * pyramid[l]->vRes - 0.5f;
			int s0 = floorf(s), t0 = floorf(t);
			float ds = s - s0, dt = t - t0;

			return (1 - ds) * (1 - dt) * texel(l, s0, t0, pyramid) +
				(1 - ds) * dt * texel(l, s0, t0 + 1, pyramid) +
				ds * (1 - dt) * texel(l, s0 + 1, t0, pyramid) +
				ds * dt * texel(l, s0 + 1, t0 + 1, pyramid);
		}

	public:

		__host__ Mipmap(ImageWrap mode, const BlockedArray<T>* pyramid, int width, int height, int level) :
			wrapMode(mode), resolution(width, height), pyramid(pyramid), level(level)
		{}

		__device__ T lookUp(const Point2f& st, float width) const {

			float levels = level - 1 + log2(fmaxf(width, (float)1e-8));

			if (levels < 0) return triangle(0, st);
			else if (levels >= level - 1) return texel(level - 1, 0, 0);
			else {
				int iLevel = floorf(levels);
				float delta = levels - iLevel;
				return linearInterpolation(delta, triangle(iLevel, st), triangle(iLevel + 1, st));
			}
		}
	};
}

#endif