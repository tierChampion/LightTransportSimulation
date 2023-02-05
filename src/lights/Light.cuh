#ifndef __light_cuh__
#define __light_cuh__

#include "../rendering/Spectrum.cuh"
#include "../shapes/Triangle.cuh"
#include "VisibilityTester.cuh"
#include "../materials/textures/Mipmap.cuh"

namespace lts {

	class VisibilityTester;

	enum class LightFlags : int {
		DeltaPosition = 1, DeltaDirection = 2, Area = 4, Infinite = 8
	};

	__device__ inline bool isDeltaLight(int flags) {
		return flags & (int)LightFlags::DeltaPosition ||
			flags & (int)LightFlags::DeltaDirection;
	}

	class Light {

	public:

		const int flag;
		Transform lightToWorld, worldToLight;

		__device__ Light(const Transform& LTW, int flag) : lightToWorld(LTW),
			worldToLight(LTW.getInverse()), flag(flag)
		{}
		__device__ virtual Spectrum sampleLi(const Interaction& it, const Point2f& sample, Vector3f* wi, float* pdf,
			VisibilityTester* vis) const {
			printf("Warning: called sampleLi in the regular Light class. The method thus does nothing.\n");
			return Spectrum(0.0f);
		}

		__device__ virtual Spectrum power() const = 0;

		__device__ virtual Spectrum Le(const Ray& r) const { return Spectrum(0.0f); }

		__device__ virtual Spectrum sampleLe(const Point2f& u1, const Point2f& u2,
			Ray* ray, Normal3f* nLight, float* pdfPos, float* pdfDir) const {
			return Spectrum(0.0f);
		}

		__device__ virtual float PdfLi(const Interaction& it, const Vector3f& wi) const = 0;

		__device__ virtual void PdfLe(const Ray& ray, const Normal3f& nLight,
			float* pdfPos, float* pdfDir) const = 0;
	};

	class PointLight : public Light {

		Point3f pLight;
		Spectrum I; // power per unit solid angle

	public:

		__device__ PointLight(const Transform& LTW, const Spectrum& I) : Light(LTW, (int)LightFlags::DeltaPosition),
			I(I), pLight(LTW(Point3f(0.0f, 0.0f, 0.0f))) {}

		__device__ Spectrum L(const Interaction& it, const Vector3f& w) const {
			return Spectrum(0.0f);
		}

		__device__ Spectrum sampleLi(const Interaction& it, const Point2f& sample, Vector3f* wi, float* pdf,
			VisibilityTester* vis) const override;

		__device__ Spectrum sampleLe(const Point2f& u1, const Point2f& u2,
			Ray* ray, Normal3f* nLight, float* pdfPos, float* pdfDir) const override;

		__device__ float PdfLi(const Interaction& it, const Vector3f& wi) const override {
			return 0.0f;
		}

		__device__ void PdfLe(const Ray& ray, const Normal3f& nLight,
			float* pdfPos, float* pdfDir) const override {
			*pdfPos = 0.0f;
			*pdfDir = uniformSampleSpherePDF();
		}

		__device__ Spectrum power() const override {
			return 4 * M_PI * I;
		}
	};

	class AreaLight : public Light {

		const Spectrum Lemit;
		const Triangle* tri;
		const float area;

	public:

		__device__ AreaLight(const Transform& LTW, const Spectrum& Le, const Triangle* tri) :
			Light(LTW, (int)LightFlags::Area), tri(tri), Lemit(Le), area(tri->area())
		{}

		__device__ Spectrum L(const Interaction& it, const Vector3f& w) const {
			return dot(it.n, w) > 0.f ? Lemit : Spectrum(0.0f);
		}

		__device__ Spectrum power() const override {
			return Lemit * area * M_PI;
		}

		__device__ Spectrum sampleLi(const Interaction& it, const Point2f& sample, Vector3f* wi, float* pdf,
			VisibilityTester* vis) const override;

		__device__ Spectrum sampleLe(const Point2f& u1, const Point2f& u2,
			Ray* ray, Normal3f* nLight, float* pdfPos, float* pdfDir) const override;

		__device__ float PdfLi(const Interaction& it, const Vector3f& wi) const override {
			return tri->Pdf(it, wi);
		}

		__device__ void PdfLe(const Ray& ray, const Normal3f& nLight,
			float* pdfPos, float* pdfDir) const override;
	};

	class InfiniteLight : public Light {

		Spectrum Lmap; // maybe switch to texture (no idea where to get them)
		Point3f worldCenter;
		float worldRadius;

		// le, pdfli, sampleli

	public:

		__device__ InfiniteLight(const Transform& LTW, const Spectrum& L) :
			Light(LTW, (int)LightFlags::Infinite), Lmap(L)
		{}

		__device__ void preprocess(const Bounds3f& sceneBounds) {
			worldCenter = sceneBounds.pMin * 0.5f + sceneBounds.pMax * 0.5f; // turn to function
			worldRadius = sceneBounds.diagonal().length() / 2.0f;
		}

		__device__ Spectrum power() const override {
			return M_PI * worldRadius * Lmap;
		}

		__device__ Spectrum Le(const Ray& r) const override {

			return Lmap;
		}

		__device__ Spectrum sampleLi(const Interaction& it, const Point2f& sample, Vector3f* wi, float* pdf,
			VisibilityTester* vis) const override;

		__device__ Spectrum sampleLe(const Point2f& u1, const Point2f& u2,
			Ray* ray, Normal3f* nLight, float* pdfPos, float* pdfDir) const override {
			printf("No sampleLe has been implemented for the infinite light.\n");
			return Spectrum(0.0f);
		}

		__device__ float PdfLi(const Interaction& it, const Vector3f& wi) const override {

			Vector3f w = worldToLight(wi);
			float sinTheta = sinf(sphericalTheta(w));
			return uniformSampleHemispherePDF() / (2 * M_PI * M_PI * sinTheta);
		}

		__device__ void PdfLe(const Ray& ray, const Normal3f& nLight,
			float* pdfPos, float* pdfDir) const override {
			printf("No PdfLe has been implemented for the infinite light.\n");
		}
	};
}

#endif