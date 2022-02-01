#ifndef __light_cuh__
#define __light_cuh__

#include "../rendering/Spectrum.cuh"
#include "../shapes/Triangle.cuh"
#include "VisibilityTester.cuh"

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

	protected:

		const int nSamples;

	public:

		const int flag;
		Transform lightToWorld, worldToLight;

		__device__ Light(const Transform& LTW, int flag, int nSamples = 1) : lightToWorld(LTW),
			worldToLight(LTW.getInverse()), flag(flag), nSamples((int)fmaxf(1, nSamples))
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

		__device__ virtual float PdfLe(const Ray& ray, const Normal3f& nLight,
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
			VisibilityTester* vis) const override {

			*wi = normalize(pLight - it.p);
			*pdf = 1.0f;
			*vis = VisibilityTester(it, Interaction(pLight));
			return I / distanceSquared(pLight, it.p);
		}

		__device__ Spectrum sampleLe(const Point2f& u1, const Point2f& u2,
			Ray* ray, Normal3f* nLight, float* pdfPos, float* pdfDir) const override {

			*ray = Ray(pLight, uniformSampleSphere(u1), INFINITY);
			*nLight = (Normal3f)ray->d;

			*pdfPos = 1;
			*pdfDir = uniformSampleSpherePDF();
			return I;
		}

		__device__ float PdfLi(const Interaction& it, const Vector3f& wi) const override {
			return 0.0f;
		}

		__device__ float PdfLe(const Ray& ray, const Normal3f& nLight,
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

		__device__ AreaLight(const Transform& LTW, const Spectrum& Le, int nSamples, const Triangle* tri) :
			Light(LTW, (int)LightFlags::Area, nSamples), tri(tri), Lemit(Le), area(tri->area())
		{}

		__device__ Spectrum L(const Interaction& it, const Vector3f& w) const {
			return dot(it.n, w) > 0.f ? Lemit : Spectrum(0.0f);
		}

		__device__ Spectrum power() const override {
			return Lemit * area * M_PI;
		}

		__device__ Spectrum sampleLi(const Interaction& it, const Point2f& sample, Vector3f* wi, float* pdf,
			VisibilityTester* vis) const override {

			Interaction pShape = tri->sample(it, sample);

			*wi = normalize(pShape.p - it.p);
			*pdf = tri->Pdf(it, *wi);
			*vis = VisibilityTester(it, pShape);
			return L(pShape, -*wi);
		}

		__device__ Spectrum sampleLe(const Point2f& u1, const Point2f& u2,
			Ray* ray, Normal3f* nLight, float* pdfPos, float* pdfDir) const override {

			// Sample position
			Interaction pShape = tri->sample(u1);
			*pdfPos = tri->Pdf(pShape);
			*nLight = pShape.n;

			// Sample direction
			Vector3f w = cosineSampleHemisphere(u2);
			*pdfDir = cosineSampleHemispherePDF(w.z);
			Vector3f v1, v2, n(pShape.n);
			coordinateSystem(n, &v1, &v2);
			w = v1 * w.x + v2 * w.y + n * w.z;

			*ray = pShape.spawnRay(w);
			return L(pShape, w);
		}

		__device__ float PdfLi(const Interaction& it, const Vector3f& wi) const override {
			return tri->Pdf(it, wi);
		}

		__device__ float PdfLe(const Ray& ray, const Normal3f& nLight,
			float* pdfPos, float* pdfDir) const override {

			Interaction pShape(ray.o, nLight, Vector3f(), Vector3f(nLight));

			*pdfPos = tri->Pdf(pShape);

			*pdfDir = cosineSampleHemispherePDF(dot(nLight, ray.d));
		}
	};
}

#endif