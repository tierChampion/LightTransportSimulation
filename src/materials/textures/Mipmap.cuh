#ifndef __mipmap_cuh__
#define __mipmap_cuh__

#include "../../core/MemoryHelper.cuh"
#include "../../geometry/Geometry.cuh"
#include "../../rendering/Spectrum.cuh"

namespace lts {

	enum class ImageWrap {
		Repeat, Black, Clamp
	};

	__device__ inline Spectrum texel(int s, int t, int l, const BlockedArray<Spectrum>* pyramid, ImageWrap wrapMode) {

		const BlockedArray<Spectrum>& level = pyramid[l];

		switch (wrapMode) {
		case ImageWrap::Repeat: s = fmodf(s, level.getUResolution());
			t = fmodf(t, level.getVResolution()); break;
		case ImageWrap::Clamp: s = clamp(s, 0, level.getUResolution() - 1);
			t = clamp(t, 0, level.getVResolution() - 1); break;
		case ImageWrap::Black: {
			if (s < 0 || s >= level.getUResolution() ||
				t < 0 || t >= level.getVResolution()) return Spectrum(0.0f);
			break;
		}
		}


		return level(s, t);
	}

	class Mipmap {

		ImageWrap wrapMode;
		Point2i resolution;
		const BlockedArray<Spectrum>* pyramid;
		int level;

		__device__ Spectrum triangle(int l, const Point2f& st) const {
			l = clamp(l, 0, level - 1);
			float s = st[0] * pyramid[l].getUResolution() - 0.5f;
			float t = st[1] * pyramid[l].getVResolution() - 0.5f;
			int s0 = floorf(s), t0 = floorf(t);
			float ds = s - s0, dt = t - t0;

			return (1 - ds) * (1 - dt) * texel(s0, t0, l, pyramid, wrapMode) +
				(1 - ds) * dt * texel(s0, t0 + 1, l, pyramid, wrapMode) +
				ds * (1 - dt) * texel(s0 + 1, t0, l, pyramid, wrapMode) +
				ds * dt * texel(s0 + 1, t0 + 1, l, pyramid, wrapMode);
		}

	public:

		__host__ Mipmap() : wrapMode(ImageWrap::Black), level(1) {}
		__host__ Mipmap(ImageWrap mode, const BlockedArray<Spectrum>* pyramid, int width, int height, int level) :
			wrapMode(mode), resolution(width, height), pyramid(pyramid), level(level)
		{}

		__device__ Spectrum lookUp(const Point2f& st, float width) const {

			float levels = level - 1 + log2(fmaxf(width, (float)1e-8));

			if (levels < 0) return triangle(0, st);
			else if (levels >= level - 1) return texel(level - 1, 0, 0, pyramid, wrapMode);
			else {
				int iLevel = floorf(levels);
				float delta = levels - iLevel;
				return linearInterpolation(delta, triangle(iLevel, st), triangle(iLevel + 1, st));
			}
		}
	};

	__global__ void pyramidBaseInit(BlockedArray<Spectrum>* pyramid, int width, int height, Spectrum* image, int levels);

	__global__ void pyramidInitKernel(BlockedArray<Spectrum>* pyramid, int level, ImageWrap mode);

	__host__ Mipmap CreateMipMap(const char* file, ImageWrap wrapMode);
}

#endif