#ifndef __specular_cuh__
#define __specular_cuh__

#include "BxDF.cuh"

namespace lts {

	class SpecularReflection : public BxDF {

		const Spectrum R;
		Fresnel* fresnel;

	public:

		__device__ SpecularReflection(const Spectrum& R, Fresnel* fresnel) :
			BxDF(BxDFType(BSDF_REFLECTION | BSDF_SPECULAR)), R(R), fresnel(fresnel) {}

		__device__ void clean() override {
			free(fresnel);
		}

		__device__ Spectrum f(const Vector3f& wo, const Vector3f& wi) const override {
			return Spectrum(0.0f);
		}

		__device__ float Pdf(const Vector3f& wo, const Vector3f& wi) const override {
			return 0.0f;
		}

		__device__ virtual Spectrum sampleF(const Vector3f& wo, Vector3f* wi,
			Point2f& sample, float* pdf, BxDFType* sampledType = nullptr) const override {

			// Perfect specular reflection
			*wi = Vector3f(-wo.x, -wo.y, wo.z);

			*pdf = 1.0f;

			return fresnel->evaluate(cosTheta(*wi)) * R / absCosTheta(*wi);
		}
	};

	class SpecularTransmission : public BxDF {

		const Spectrum T;
		const float etaA, etaB;
		const FresnelDielectric fresnel;
		const TransportMode mode;

	public:

		__device__ SpecularTransmission(const Spectrum& T, float etaA, float etaB,
			TransportMode mode) :
			BxDF(BxDFType(BSDF_TRANSMISSION | BSDF_REFLECTION)), T(T),
			etaA(etaA), etaB(etaB), fresnel(etaA, etaB), mode(mode) {}

		__device__ Spectrum f(const Vector3f& wo, const Vector3f& wi) const override {
			return Spectrum(0.0f);
		}

		__device__ float Pdf(const Vector3f& wo, const Vector3f& wi) const override {
			return 0.0f;
		}

		__device__ virtual Spectrum sampleF(const Vector3f& wo, Vector3f* wi,
			Point2f& sample, float* pdf, BxDFType* sampledType = nullptr) const override {

			bool entering = cosTheta(wo) > 0;
			float etaI = entering ? etaA : etaB;
			float etaT = entering ? etaB : etaA;

			if (!refract(wo, faceTowards(Normal3f(0, 0, 1), wo), etaI / etaT, wi)) return 0.0f;

			*pdf = 1.0f;
			Spectrum ft = T * (Spectrum(1.0f) - fresnel.evaluate(cosTheta(*wi)));

			// Scale rays starting from light sources
			if (mode == TransportMode::Radiance)
				ft *= (etaI * etaI) / (etaT * etaT);

			return ft / absCosTheta(*wi);
		}
	};

	class FresnelSpecular : public BxDF {

		const Spectrum R, T;
		const float etaA, etaB;
		const FresnelDielectric fresnel;
		const TransportMode mode;

	public:

		__device__ FresnelSpecular(const Spectrum& R, const Spectrum& T, float etaA, float etaB,
			TransportMode mode) :
			BxDF(BxDFType(BSDF_REFLECTION | BSDF_TRANSMISSION | BSDF_SPECULAR)),
			R(R), T(T), etaA(etaA), etaB(etaB), fresnel(etaA, etaB), mode(mode)
		{}

		__device__ Spectrum f(const Vector3f& wo, const Vector3f& wi) const override {
			return Spectrum(0.0f);
		}

		__device__ float Pdf(const Vector3f& wo, const Vector3f& wi) const override {
			return 0.0f;
		}

		__device__ virtual Spectrum sampleF(const Vector3f& wo, Vector3f* wi,
			Point2f& sample, float* pdf, BxDFType* sampledType = nullptr) const override {

			float f = fresnelReflectionDielectric(cosTheta(wo), etaA, etaB);
			if (sample[0] < f) {
				*wi = Vector3f(-wo.x, -wo.y, -wo.z);
				if (sampledType) *sampledType = BxDFType(BSDF_SPECULAR | BSDF_REFLECTION);
				*pdf = f;
				return f * R / absCosTheta(*wi);
			}
			else {
				bool entering = cosTheta(wo) > 0;
				float etaI = entering ? etaA : etaB;
				float etaT = entering ? etaB : etaA;

				if (!refract(wo, faceTowards(Normal3f(0, 0, 1), wo), etaI / etaT, wi)) return 0.0f;

				Spectrum ft = T * (1 - f);

				// Scale rays starting from light sources
				if (mode == TransportMode::Radiance)
					ft *= (etaI * etaI) / (etaT * etaT);

				if (sampledType) *sampledType = BxDFType(BSDF_SPECULAR | BSDF_TRANSMISSION);

				*pdf = 1.0f - f;
				return ft / absCosTheta(*wi);
			}
		}
	};
}

#endif