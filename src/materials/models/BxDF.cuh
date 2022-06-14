#ifndef __bxdf_cuh__
#define __bxdf_cuh__

#include "../../rendering/Spectrum.cuh"
#include "../../geometry/Geometry.cuh"
#include "../../sampling/Sampling.cuh"
#include "../TransportMode.cuh"

namespace lts {

	/**
	* Computes the fresnel reflectance of unpolarized light at the boundary of two dielectric medium.
	* @param cosThetaI - cosine of the incident direction
	* @param etaI and etaT - indices of refraction for the incident and transmitted medium.
	*/
	__device__ inline float fresnelReflectionDielectric(float cosThetaI, float etaI, float etaT) {
		cosThetaI = clamp(cosThetaI, 1, -1);
		bool entering = cosThetaI > 0.0f;
		if (!entering) {
			float temp = etaI;
			etaI = etaT;
			etaT = temp;
			cosThetaI = fabsf(cosThetaI);
		}

		float sinThetaI = sqrtf(fmaxf(0.0f, 1 - cosThetaI * cosThetaI));
		float sinThetaT = etaI / etaT * sinThetaI;
		if (sinThetaT >= 1) return 1;

		float cosThetaT = sqrtf(fmaxf(0.0f, 1 - sinThetaT * sinThetaT));

		float rpar1 = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
			((etaT * cosThetaI) + (etaI * cosThetaT));
		float rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
			((etaI * cosThetaI) + (etaT * cosThetaT));
		return (rpar1 * rpar1 + rperp * rperp) / 2;
	}

	/**
	* Computes the fresnel reflectance of unpolarized light at the boundary of a dielectrum media and a conductor.
	* @param cosThetaI - cosine of the incident direction
	* @param etaI and etaT - indices of refraction for the incident and transmitted medium.
	*/
	__device__ inline Spectrum fresnelReflectionConductor(float cosThetaI,
		const Spectrum& etaI, const Spectrum& etaT, const Spectrum& k) {
		cosThetaI = clamp(cosThetaI, -1, 1);
		Spectrum eta = etaT / etaI;
		Spectrum etaK = k / etaI;

		float cosThetaI2 = cosThetaI * cosThetaI;
		float sinThetaI2 = 1.0f - cosThetaI2;
		Spectrum eta2 = eta * eta;
		Spectrum etaK2 = etaK * etaK;

		Spectrum t0 = eta2 - etaK2 - sinThetaI2;
		Spectrum a2plusb2 = sqrt(t0 * t0 + 4 * eta2 * etaK2);
		Spectrum t1 = a2plusb2 + cosThetaI2;
		Spectrum a = sqrt(0.5f * (a2plusb2 + t0));
		Spectrum t2 = 2.0f * cosThetaI * a;
		Spectrum Rs = (t1 - t2) / (t1 + t2);

		Spectrum t3 = cosThetaI2 * a2plusb2 + sinThetaI2 * sinThetaI2;
		Spectrum t4 = t2 * sinThetaI2;
		Spectrum Rp = Rs * (t3 - t4) / (t3 + t4);

		return (Rp + Rs) * 0.5f;
	}

	/* Fresnel */

	class Fresnel {
	public:

		__device__ virtual ~Fresnel() {}
		__device__ virtual Spectrum evaluate(float cosThetaI) const = 0;
	};

	class FresnelConductor : public Fresnel {

		Spectrum etaI, etaT, k;

	public:

		__device__ FresnelConductor(const Spectrum& etaI, const Spectrum& etaT, const Spectrum& k) :
			etaI(etaI), etaT(etaT), k(k) {}

		__device__ Spectrum evaluate(float cosThetaI) const override {
			return fresnelReflectionConductor(fabsf(cosThetaI), etaI, etaT, k);
		}
	};

	class FresnelDielectric : public Fresnel {

		float etaI, etaT;

	public:

		__device__ FresnelDielectric(float etaI, float etaT) : etaI(etaI), etaT(etaT) {}

		__device__ Spectrum evaluate(float cosThetaI) const override {
			return fresnelReflectionDielectric(cosThetaI, etaI, etaT);
		}
	};

	class FresnelNoOp : public Fresnel {

	public:

		__device__ Spectrum evaluate(float cosThetaI) const override {
			return Spectrum(1.0f);
		}
	};

	/* BxDF */

	enum BxDFType {
		BSDF_REFLECTION = 1 << 0,
		BSDF_TRANSMISSION = 1 << 1,
		BSDF_DIFFUSE = 1 << 2,
		BSDF_GLOSSY = 1 << 3,
		BSDF_SPECULAR = 1 << 4,
		BSDF_ALL = BSDF_DIFFUSE | BSDF_GLOSSY | BSDF_SPECULAR |
		BSDF_REFLECTION | BSDF_TRANSMISSION,
	};

	__device__ inline bool sameHemisphere(const Vector3f& w, const Vector3f& wp) {
		return w.z * wp.z > 0;
	}

	__device__ inline Vector3f reflect(const Vector3f& wo, const Vector3f& n) {
		return -wo + n * (2 * dot(wo, n));
	}

	__device__ inline float cosTheta(const Vector3f& w) { return w.z; }
	__device__ inline float absCosTheta(const Vector3f& w) { return fabsf(w.z); }
	__device__ inline float cos2Theta(const Vector3f& w) { return w.z * w.z; }

	__device__ inline float sin2Theta(const Vector3f& w) { return fmaxf(1 - cos2Theta(w), 0.0f); }
	__device__ inline float sinTheta(const Vector3f& w) { return sqrtf(sin2Theta(w)); }

	__device__ inline float tanTheta(const Vector3f& w) { return sinTheta(w) / cosTheta(w); }
	__device__ inline float tan2Theta(const Vector3f& w) { return sin2Theta(w) / cos2Theta(w); }

	__device__ inline float cosPhi(const Vector3f& w) {
		float sin = sinTheta(w);
		return (sinTheta == 0) ? 1 : clamp(w.x / sin, -1, 1);
	}
	__device__ inline float cos2Phi(const Vector3f& w) { return cosPhi(w) * cosPhi(w); }

	__device__ inline float sinPhi(const Vector3f& w) {
		float sin = sinTheta(w);
		return (sinTheta == 0) ? 0 : clamp(w.y / sin, -1, 1);
	}
	__device__ inline float sin2Phi(const Vector3f& w) { return sinPhi(w) * sinPhi(w); }

	__device__ inline bool refract(const Vector3f& wi, const Normal3f& n, float eta, Vector3f* wt) {
		float cosThetaI = dot(n, wi);
		float sin2ThetaI = fmaxf(0.0f, 1.0f - cosThetaI * cosThetaI);
		float sin2ThetaT = eta * eta * sin2ThetaI;
		if (sin2ThetaT >= 1) return false;
		float cosThetaT = sqrtf(1 - sin2ThetaT);

		*wt = -wi * eta + Vector3f(n) * (eta * cosThetaI - cosThetaT);

		return true;
	}

	class BxDF {

	public:

		BxDFType type;

		__device__ virtual ~BxDF() {}
		__device__ BxDF(BxDFType type) : type(type) {}

		__device__ virtual void clean() {}

		__device__ bool matchesFlag(BxDFType t) const {
			return (type & t) == type;
		}

		__device__ virtual Spectrum f(const Vector3f& wo, const Vector3f& wi) const = 0;
		__device__ virtual Spectrum rho(const Vector3f& wo, int nSamples,
			const Point2f* samples) const {
			return Spectrum(0.0f);
		}
		__device__ virtual Spectrum rho(int nSamples,
			const Point2f* samples1, const Point2f* samples2) const {
			return Spectrum(0.0f);
		}
		__device__ virtual float Pdf(const Vector3f& wo, const Vector3f& wi) const {
			return sameHemisphere(wo, wi) ? absCosTheta(wi) * M_1_PI : 0;
		}

		__device__ virtual Spectrum sampleF(const Vector3f& wo, Vector3f* wi,
			Point2f& sample, float* pdf, BxDFType* sampledType = nullptr) const {

			*wi = cosineSampleHemisphere(sample);
			if (wo.z < 0) wi->z *= -1;

			*pdf = Pdf(wo, *wi);
			return f(wo, *wi);
		}
	};

}

#endif