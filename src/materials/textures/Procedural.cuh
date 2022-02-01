#ifndef __procedural_cuh__
#define __procedural_cuh__

#include "Texture.cuh"

namespace lts {

	class UVTexture : public Texture<Spectrum> {

		const TextureMapping2D* mapping;

	public:

		__device__ UVTexture(TextureMapping2D* m) : mapping(m) {}

		__device__ Spectrum evaluate(const SurfaceInteraction& it) const override {

			Vector2f dstdx, dstdy;
			Point2f st = mapping->map(it, &dstdx, &dstdy);

			return Spectrum(st[0] - floorf(st[0]), st[1] - floorf(st[1]), 0.0f);
		}

	};

	template <typename T>
	class CheckerboardTexture2D : public Texture<T> {

		const TextureMapping2D* mapping;
		Texture<T>* tex1, * tex2;

	public:

		__device__ CheckerboardTexture2D(TextureMapping2D* mapping, Texture<T>* t1, Texture<T>* t2) :
			mapping(mapping), tex1(t1), tex2(t2) {}

		__device__ T evaluate(const SurfaceInteraction& it) const override {

			Vector2f dstdx, dstdy;
			Point2f st = mapping->map(it, &dstdx, &dstdy);

			// no aliasing for now...
			if (((int)floorf(st[0]) + (int)floorf(st[1])) % 2 == 0) {
				return tex1->evaluate(it);
			}
			else {
				return tex2->evaluate(it);
			}
		}
	};

	// 3d checkerboard
	template <typename T>
	class CheckerboardTexture3D : public Texture<T> {

		const TextureMapping3D* mapping;
		Texture<T>* tex1, * tex2;

	public:

		__device__ CheckerboardTexture3D(TextureMapping3D* mapping, Texture<T>* t1, Texture<T>* t2) :
			mapping(mapping), tex1(t1), tex2(t2) {}

		__device__ T evaluate(const SurfaceInteraction& it) const override {

			Vector3f dpdx, dpdy;
			Point3f st = mapping->map(it, &dpdx, &dpdy);

			if (((int)floorf(st[0]) + (int)floorf(st[1]) + (int)floorf(st[2])) % 2 == 0) {
				return tex1->evaluate(it);
			}
			else
				return tex2->evaluate(it);
		}

	};

	// noise textures
	__device__ const int noisePermSize = 256;
	__device__ const int noisePerm[noisePermSize * 2] = {
	151, 160, 137, 91, 90, 15,
	131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23,
	  190,  6, 148, 247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32, 57, 177, 33,
	  88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168,  68, 175, 74, 165, 71, 134, 139, 48, 27, 166,
	  77, 146, 158, 231, 83, 111, 229, 122, 60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244,
	  102, 143, 54,  65, 25, 63, 161,  1, 216, 80, 73, 209, 76, 132, 187, 208,  89, 18, 169, 200, 196,
	  135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186,  3, 64, 52, 217, 226, 250, 124, 123,
	  5, 202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42,
	  223, 183, 170, 213, 119, 248, 152,  2, 44, 154, 163,  70, 221, 153, 101, 155, 167,  43, 172, 9,
	  129, 22, 39, 253,  19, 98, 108, 110, 79, 113, 224, 232, 178, 185,  112, 104, 218, 246, 97, 228,
	  251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241,  81, 51, 145, 235, 249, 14, 239, 107,
	  49, 192, 214,  31, 181, 199, 106, 157, 184,  84, 204, 176, 115, 121, 50, 45, 127,  4, 150, 254,
	  138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180,
	  151, 160, 137, 91, 90, 15,
	  131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23,
	  190,  6, 148, 247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32, 57, 177, 33,
	  88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168,  68, 175, 74, 165, 71, 134, 139, 48, 27, 166,
	  77, 146, 158, 231, 83, 111, 229, 122, 60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244,
	  102, 143, 54,  65, 25, 63, 161,  1, 216, 80, 73, 209, 76, 132, 187, 208,  89, 18, 169, 200, 196,
	  135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186,  3, 64, 52, 217, 226, 250, 124, 123,
	  5, 202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42,
	  223, 183, 170, 213, 119, 248, 152,  2, 44, 154, 163,  70, 221, 153, 101, 155, 167,  43, 172, 9,
	  129, 22, 39, 253,  19, 98, 108, 110, 79, 113, 224, 232, 178, 185,  112, 104, 218, 246, 97, 228,
	  251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241,  81, 51, 145, 235, 249, 14, 239, 107,
	  49, 192, 214,  31, 181, 199, 106, 157, 184,  84, 204, 176, 115, 121, 50, 45, 127,  4, 150, 254,
	  138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180
	};

	__device__ inline float noiseWeight(float t) {
		float t3 = t * t * t;
		float t4 = t3 * t;
		return 6 * t4 * t - 15 * t4 + 10 * t3;
	}

	__device__ inline float smoothStep(float min, float max, float val) {
		float v = clamp((val - min) / (max - min), 0, 1);
		return v * v * (-2 * v + 3);
	}

	__device__ inline float grad(int x, int y, int z, float dx, float dy, float dz) {

		int h = noisePerm[noisePerm[noisePerm[x] + y] + z];
		h &= 15;

		float u = h < 8 || h == 12 || h == 13 ? dx : dy;
		float v = h < 4 || h == 12 || h == 13 ? dy : dz;
		return ((h & 1) ? -u : u) + ((h & 2) ? -v : v);
	}

	__device__ inline float perlinNoise(float x, float y, float z) {

		int ix = (int)floorf(x), iy = (int)floorf(y), iz = (int)floorf(z);
		float dx = x - ix, dy = y - iy, dz = z - iz;

		ix &= noisePermSize - 1;
		iy &= noisePermSize - 1;
		iz &= noisePermSize - 1;
		float w000 = grad(ix, iy, iz, dx, dy, dz);
		float w100 = grad(ix + 1, iy, iz, dx - 1, dy, dz);
		float w010 = grad(ix, iy + 1, iz, dx, dy - 1, dz);
		float w001 = grad(ix, iy, iz + 1, dx, dy, dz - 1);
		float w110 = grad(ix + 1, iy + 1, iz, dx - 1, dy - 1, dz);
		float w011 = grad(ix, iy + 1, iz + 1, dx, dy - 1, dz - 1);
		float w101 = grad(ix + 1, iy, iz + 1, dx - 1, dy, dz - 1);
		float w111 = grad(ix + 1, iy + 1, iz + 1, dx - 1, dy - 1, dz - 1);

		float wx = noiseWeight(dx), wy = noiseWeight(dy), wz = noiseWeight(dz);
		float x00 = linearInterpolation(wx, w000, w100);
		float x10 = linearInterpolation(wx, w010, w110);
		float x01 = linearInterpolation(wx, w001, w101);
		float x11 = linearInterpolation(wx, w011, w111);
		float y0 = linearInterpolation(wy, x00, x10);
		float y1 = linearInterpolation(wy, x01, x11);
		return linearInterpolation(wz, y0, y1);
	}

	__device__ inline float perlinNoise(const Point3f& p) {
		return perlinNoise(p.x, p.y, p.z);
	}

	__device__ inline float fbm(const Point3f& p, const Vector3f& dpdx, const Vector3f& dpdy,
		float omega, int maxOctaves) {

		float len2 = fmaxf(dpdx.lengthSquared(), dpdy.lengthSquared());
		float n = clamp(-1 - 0.5f * log2f(len2), 0, maxOctaves);
		int nInt = floorf(n);

		float sum = 0, lambda = 1, o = 1;
		for (int i = 0; i < nInt; i++) {
			sum += o * perlinNoise(p * lambda);
			lambda *= 1.99f;
			o *= omega;
		}

		float nPartial = n - nInt;
		sum += o * smoothStep(0.3f, 0.7f, nPartial) * perlinNoise(p * lambda);

		return sum;
	}

	__device__ inline float turbulence(const Point3f& p, const Vector3f& dpdx, const Vector3f& dpdy,
		float omega, int maxOctaves) {

		float len2 = fmaxf(dpdx.lengthSquared(), dpdy.lengthSquared());
		float n = clamp(-1 - 0.5f * log2f(len2), 0, maxOctaves);
		int nInt = floorf(n);

		float sum = 0, lambda = 1, o = 1;
		for (int i = 0; i < nInt; i++) {
			sum += o * fabsf(perlinNoise(p * lambda));
			lambda *= 1.99f;
			o *= omega;
		}

		float nPartial = n - nInt;
		sum += o * linearInterpolation(smoothStep(0.3f, 0.7f, nPartial),
			0.2f, fabsf(perlinNoise(p * lambda)));

		for (int i = nInt; i < maxOctaves; i++) {
			sum += o * 0.2f;
			o *= omega;
		}

		return sum;
	}

	template <typename T>
	class FBmTexture : public Texture<T> {

		const TextureMapping3D mapping;
		const float omega;
		const int octaves;

	public:

		__device__ FBmTexture(const Transform& WTT, int octaves, float omega) : mapping(WTT),
			octaves(octaves), omega(omega) {}

		__device__ T evaluate(const SurfaceInteraction& it) const override {

			Vector3f dpdx, dpdy;
			Point3f p = mapping.map(it, &dpdx, &dpdy);
			return fbm(p, dpdx, dpdy, omega, octaves);
		}
	};

	template <typename T>
	class WrinkledTexture : public Texture<T> {

		const TextureMapping3D mapping;
		const float omega;
		const int octaves;

	public:

		__device__ WrinkledTexture(const Transform& WTT, int octaves, float omega) : mapping(WTT),
			octaves(octaves), omega(omega) {}

		__device__ T evaluate(const SurfaceInteraction& it) const override {

			Vector3f dpdx, dpdy;
			Point3f p = mapping.map(it, &dpdx, &dpdy);
			return turbulence(p, dpdx, dpdy, omega, octaves);
		}
	};

	template <typename T>
	class WindyTexture : public Texture<T> {

		const TextureMapping3D mapping;

	public:

		__device__ WindyTexture(const Transform& WTT) : mapping(WTT) {}

		__device__ T evaluate(const SurfaceInteraction& it) const override {

			Vector3f dpdx, dpdy;
			Point3f p = mapping.map(it, &dpdx, &dpdy);

			float windStrength = fbm(p * 0.1f, dpdx * 0.1f, dpdy * 0.1f, 0.5f, 3.0f);
			float waveHeight = fbm(p, dpdx, dpdy, 0.5f, 6);
			return fabsf(windStrength) * waveHeight;
		}
	};

	class MarbleTexture : public Texture<Spectrum> {

		const TextureMapping3D mapping;
		const int octaves;
		const float omega, scale, variation;

	public:

		__device__ MarbleTexture(const Transform& WTT, int octaves, float omega, float scale, float variation) :
			mapping(WTT), octaves(octaves), omega(omega), scale(scale), variation(variation) {}

		__device__ Spectrum evaluate(const SurfaceInteraction& it) const override {

			Vector3f dpdx, dpdy;
			Point3f p = mapping.map(it, &dpdx, &dpdy);
			p *= scale;
			float marble = p.y + variation * fbm(p, dpdx * scale, dpdy * scale, omega, octaves);
			float t = 0.5f + 0.5f * sinf(marble);
			// bunch of stuff to estimate color of real marble
			static float c[][3] = {
				{ .58f, .58f, .6f }, { .58f, .58f, .6f }, { .58f, .58f, .6f },
		   { .5f, .5f, .5f }, { .6f, .59f, .58f }, { .58f, .58f, .6f },
		   { .58f, .58f, .6f }, {.2f, .2f, .33f }, { .58f, .58f, .6f },
			};

#define NC sizeof(c) / sizeof(c[0])
#define NSEG (NC - 3)

			int first = (int)floorf(t * NSEG);
			t = (t * NSEG - first);

			Spectrum c0 = Spectrum(c[first]);
			Spectrum c1 = Spectrum(c[first + 1]);
			Spectrum c2 = Spectrum(c[first + 2]);
			Spectrum c3 = Spectrum(c[first + 3]);

			Spectrum s0 = (1.0f - t) * c0 + t * c1;
			Spectrum s1 = (1.0f - t) * c1 + t * c2;
			Spectrum s2 = (1.0f - t) * c2 + t * c3;
			s0 = (1.0f - t) * s0 + t * s1;
			s1 = (1.0f - t) * s1 + t * s2;

			return 1.5f * ((1.0f - t) * s0 + t * s1);
		}
	};
}

#endif