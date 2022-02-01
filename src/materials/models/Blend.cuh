#ifndef __blend_cuh__
#define __blend_cuh__

#include "Microfacet.cuh"

namespace lts {

	// remove pointer to distribution

	class FresnelBlend : public BxDF {

		const Spectrum Rd, Rs; // diffuse, specular
		const TrowbridgeReitzMicrofacetDistribution distribution;

	public:

		__device__ FresnelBlend(const Spectrum& Rd, const Spectrum& Rs,
			float alphaX, float alphaY) : BxDF(BxDFType(BSDF_REFLECTION | BSDF_GLOSSY)),
			Rd(Rd), Rs(Rs), distribution(alphaX, alphaY) {}

		__device__ Spectrum schlinkFresnel(float cosTheta) const {
			auto pow5 = [](float v) {return (v * v) * (v * v) * v; };
			return Rs + pow5(1 - cosTheta) * (Spectrum(1.0f) - Rs);
		}

		__device__ Spectrum f(const Vector3f& wo, const Vector3f& wi) const override {
			auto pow5 = [](float v) {return (v * v) * (v * v) * v; };

			Spectrum diffuse = (28.0f / (23.0f * M_PI)) * Rd *
				(Spectrum(1.0f) - Rs) *
				(1 - pow5(1 - 0.5f * absCosTheta(wi))) *
				(1 - pow5(1 - 0.5f * absCosTheta(wo)));

			Vector3f wh = wi + wo;
			if (wh.x == 0 && wh.y == 0 && wh.z == 0) return Spectrum(0.0f);
			wh = normalize(wh);
			Spectrum specular = distribution.d(wh) /
				(4 * absDot(wi, wh) *
					fmaxf(absCosTheta(wi), absCosTheta(wo))) * schlinkFresnel(dot(wi, wh));
			return diffuse + specular;
		}

		__device__ Spectrum sampleF(const Vector3f& wo, Vector3f* wi,
			Point2f& sample, float* pdf, BxDFType* sampledType = nullptr) const override {

			if (sample[0] < 0.5f) {
				sample[0] *= 2;
				*wi = cosineSampleHemisphere(sample);
				if (wo.z < 0) wi->z *= -1;
			}
			else {
				sample[0] = 2 * (sample[0] - 0.5f);
				Vector3f wh = distribution.sampleWH(wo, sample);
				*wi = reflect(wo, wh);
				if (!sameHemisphere(wo, *wi)) return Spectrum(0.0f);
			}
			*pdf = Pdf(wo, *wi);
			return f(wo, *wi);
		}

		__device__ virtual float Pdf(const Vector3f& wo, const Vector3f& wi) const {

			if (!sameHemisphere(wo, wi)) return 0;

			Vector3f wh = normalize(wo + wi);
			float pdfWH = distribution.Pdf(wo, wh);
			return 0.5f * (absCosTheta(wi) * M_1_PI +
				pdfWH / (4 * dot(wo, wh)));
		}
	};

}

#endif