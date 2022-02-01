#ifndef __texture_cuh__
#define __texture_cuh__

#include "../../geometry/Interaction.cuh"
#include "Mapping.cuh"

namespace lts {

	template <typename T>
	class Texture {

	public:

		__device__ virtual ~Texture() {}
		__device__ virtual T evaluate(const SurfaceInteraction& it) const = 0;

	};

	template <typename T>
	class ConstantTexture : public Texture<T> {

		T value;

	public:

		__device__ ConstantTexture(T value) : value(value) {}

		__device__ T evaluate(const SurfaceInteraction& it) const override {
			return value;
		}
	};

	template <typename T>
	class BilerpTexture : public Texture<T> {

		const TextureMapping2D* mapping;
		T v00, v01, v10, v11;

	public:

		__device__ BilerpTexture(TextureMapping2D* m, T v00, T v01, T v10, T v11) :
			mapping(m), v00(v00), v01(v01), v10(v10), v11(v11) {}

		__device__ T evaluate(const SurfaceInteraction& it) const override {

			Vector2f dstdx, dstdy;
			Point2f st = mapping->map(it, &dstdx, &dstdy);
			return (1 - st[0]) * (1 - st[1]) * v00 + (1 - st[0]) * st[1] * v01 +
				st[0] * (1 - st[1]) * v10 + st[0] * st[1] * v11;
		}
	};
}

#endif