#ifndef __image_texture_cuh__
#define __image_texture_cuh__

#include "Texture.cuh"
#include "Mipmap.cuh"

#ifndef EVALUATION_WIDTH
#define EVALUATION_WIDTH 0.01f
#endif

namespace lts {

	class ImageTexture : public Texture<Spectrum> {

		TextureMapping2D* mapping;
		Mipmap* mpmp;

	public:

		__device__ ImageTexture(TextureMapping2D* m, Mipmap* mpmp) : mapping(m), mpmp(mpmp) {}

		__device__ Spectrum evaluate(const SurfaceInteraction& si) const override {

			Vector2f dstdx, dstdy;
			Point2f st = mapping->map(si, &dstdx, &dstdy);
			return mpmp->lookUp(st, EVALUATION_WIDTH);
		}
	};
}

#endif