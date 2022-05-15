#ifndef __light_cu__
#define __light_cu__

#include "Light.cuh"

namespace lts {

	__device__ Spectrum PointLight::sampleLi(const Interaction& it, const Point2f& sample, Vector3f* wi, float* pdf,
		VisibilityTester* vis) const {

		*wi = normalize(pLight - it.p);
		*pdf = 1.0f;
		*vis = VisibilityTester(it, Interaction(pLight));
		return I / distanceSquared(pLight, it.p);
	}

	__device__ Spectrum PointLight::sampleLe(const Point2f& u1, const Point2f& u2,
		Ray* ray, Normal3f* nLight, float* pdfPos, float* pdfDir) const {

		*ray = Ray(pLight, uniformSampleSphere(u1), INFINITY);
		*nLight = (Normal3f)ray->d;

		*pdfPos = 1;
		*pdfDir = uniformSampleSpherePDF();
		return I;
	}

	__device__ Spectrum AreaLight::sampleLi(const Interaction& it, const Point2f& sample, Vector3f* wi, float* pdf,
		VisibilityTester* vis) const {

		Interaction pShape = tri->sample(it, sample);

		*wi = normalize(pShape.p - it.p);
		*pdf = tri->Pdf(it, *wi);
		*vis = VisibilityTester(it, pShape);
		return L(pShape, -*wi);
	}

	__device__ Spectrum AreaLight::sampleLe(const Point2f& u1, const Point2f& u2,
		Ray* ray, Normal3f* nLight, float* pdfPos, float* pdfDir) const {

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

	__device__ float AreaLight::PdfLe(const Ray& ray, const Normal3f& nLight,
		float* pdfPos, float* pdfDir) const {

		Interaction pShape(ray.o, nLight, Vector3f(), Vector3f(nLight));

		*pdfPos = tri->Pdf(pShape);

		*pdfDir = cosineSampleHemispherePDF(dot(nLight, ray.d));
	}
}

#endif