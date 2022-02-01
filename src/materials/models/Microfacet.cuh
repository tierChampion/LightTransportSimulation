#ifndef __microfacet_cuh__
#define __microfacet_cuh__

#include "BxDF.cuh"

namespace lts {

	// todo sampling of the bxdfs and distributions!

	/**
	* Estimation of microfacet material for lambertian materials.
	*/
	class OrenNayar : public BxDF {

		const Spectrum R;
		float A, B;

	public:

		__device__ OrenNayar(const Spectrum& R, float sigma) :
			BxDF(BxDFType(BSDF_REFLECTION | BSDF_DIFFUSE)), R(R) {

			sigma = toRadians(sigma);
			float sigma2 = sigma * sigma;
			A = 1.0f - (sigma2 / (2.0f * (sigma2 + 0.33f)));
			B = 0.45f * sigma2 / (sigma2 + 0.09f);
		}

		__device__ Spectrum f(const Vector3f& wo, const Vector3f& wi) const override {
			float sinThetaO = sinTheta(wo);
			float sinThetaI = sinTheta(wi);

			float maxCos = 0.0f;
			if (sinThetaI > 1e-4 && sinThetaO > 1e-4) {
				float sinPhiO = sinPhi(wo), cosPhiO = cosPhi(wo);
				float sinPhiI = sinPhi(wi), cosPhiI = cosPhi(wi);
				float dCos = cosPhiI * cosPhiO + sinPhiI * sinPhiO;
				maxCos = fmaxf(0.0f, dCos);
			}

			float sinAlpha, tanBeta;
			if (absCosTheta(wi) > absCosTheta(wo)) {
				sinAlpha = sinThetaO;
				tanBeta = sinThetaI / absCosTheta(wi);
			}
			else {
				sinAlpha = sinThetaI;
				tanBeta = sinThetaO / absCosTheta(wo);
			}

			return R * M_1_PI * (A + B * maxCos * sinAlpha * tanBeta);
		}
	};

	__device__ static void TrowbridgeReitzSample11(float cosT, float U1, float U2,
		float* slopeX, float* slopeY) {
		// special case (normal incidence)
		if (cosT > .9999) {
			float r = sqrtf(U1 / (1 - U1));
			float phi = 6.28318530718 * U2;
			*slopeX = r * cos(phi);
			*slopeY = r * sin(phi);
			return;
		}

		float sinT =
			sqrtf(fmaxf(0.0f, 1.0f - cosT * cosT));
		float tanT = sinT / cosT;
		float a = 1 / tanT;
		float G1 = 2 / (1 + sqrtf(1.f + 1.f / (a * a)));

		// sample slope_x
		float A = 2 * U1 / G1 - 1;
		float tmp = 1.f / (A * A - 1.f);
		if (tmp > 1e10) tmp = 1e10;
		float B = tanT;
		float D = sqrtf(
			fmaxf(float(B * B * tmp * tmp - (A * A - B * B) * tmp), 0.0f));
		float slope_x_1 = B * tmp - D;
		float slope_x_2 = B * tmp + D;
		*slopeX = (A < 0 || slope_x_2 > 1.f / tanT) ? slope_x_1 : slope_x_2;

		// sample slope_y
		float S;
		if (U2 > 0.5f) {
			S = 1.f;
			U2 = 2.f * (U2 - .5f);
		}
		else {
			S = -1.f;
			U2 = 2.f * (.5f - U2);
		}
		float z =
			(U2 * (U2 * (U2 * 0.27385f - 0.73369f) + 0.46341f)) /
			(U2 * (U2 * (U2 * 0.093073f + 0.309420f) - 1.000000f) + 0.597999f);
		*slopeY = S * z * sqrtf(1.f + *slopeX * *slopeX);

		assert(!isinf(*slopeY));
		assert(!isnan(*slopeY));
	}

	__device__ static Vector3f TrowbridgeReitzSample(const Vector3f& wi, float alphaX,
		float alphaY, float U1, float U2) {
		// 1. stretch wi
		Vector3f wiStretched =
			normalize(Vector3f(alphaX * wi.x, alphaY * wi.y, wi.z));

		// 2. simulate P22_{wi}(x_slope, y_slope, 1, 1)
		float slope_x, slope_y;
		TrowbridgeReitzSample11(cosTheta(wiStretched), U1, U2, &slope_x, &slope_y);

		// 3. rotate
		float tmp = cosPhi(wiStretched) * slope_x - sinPhi(wiStretched) * slope_y;
		slope_y = sinPhi(wiStretched) * slope_x + cosPhi(wiStretched) * slope_y;
		slope_x = tmp;

		// 4. unstretch
		slope_x = alphaX * slope_x;
		slope_y = alphaY * slope_y;

		// 5. compute normal
		return normalize(Vector3f(-slope_x, -slope_y, 1.));
	}

	// trowbridgereitz distribution
	class TrowbridgeReitzMicrofacetDistribution {

		const float alphaX, alphaY;

		__device__ float lambda(const Vector3f& w) const {
			float absTanTheta = fabsf(tanTheta(w));
			if (isinf(absTanTheta)) return 0.0f;
			float alpha = sqrtf(cos2Phi(w) * alphaX * alphaX +
				sin2Phi(w) * alphaY * alphaY);
			float alpha2Tan2Theta = (alpha * absTanTheta) * (alpha * absTanTheta);
			return (-1 + sqrtf(1.0f + alpha2Tan2Theta)) / 2;
		}

	public:

		__device__ static float roughnessToAlpha(float roughness) {
			roughness = fmaxf(roughness, (float)1e-3);
			float x = logf(roughness);
			return 1.62142f + 0.819955f * x + 0.1734f * x * x +
				0.0171201f * x * x * x + 0.000640711f * x * x * x * x;
		}

		__device__ TrowbridgeReitzMicrofacetDistribution(float alphaX, float alphaY) :
			alphaX(alphaX), alphaY(alphaY) {}

		__device__ float d(const Vector3f& wh) const {

			float tan2 = tan2Theta(wh);
			if (isinf(tan2)) return 0;
			const float cos4 = cos2Theta(wh) * cos2Theta(wh);
			float e = (cos2Phi(wh) / (alphaX * alphaX) +
				sin2Phi(wh) / (alphaY * alphaY)) * tan2;
			return 1 / (M_PI * alphaX * alphaY * cos4 * (1 + e) * (1 + e));
		}

		__device__ float g1(const Vector3f& w) const {
			return 1 / (1 + lambda(w));
		}

		__device__ float g(const Vector3f& wo, const Vector3f& wi) const {
			return 1 / (1 + lambda(wo) + lambda(wi));
		}

		__device__ Vector3f sampleWH(const Vector3f& wo, const Point2f sample) const {

			Vector3f wh;

			bool flip = wo.z < 0;
			wh = TrowbridgeReitzSample(flip ? -wo : wo, alphaX, alphaY, sample[0], sample[1]);
			if (flip) wh = -wh;

			return wh;
		}

		__device__ float Pdf(const Vector3f& wo, const Vector3f& wh) const {
			return d(wh) * g1(wo) * absDot(wo, wh) / absCosTheta(wo);
		}
	};

	// torrance sparrow bxdf (microfacet reflection)
	class MicrofacetReflection : public BxDF {

		const Spectrum R;
		const TrowbridgeReitzMicrofacetDistribution distribution;
		Fresnel* fresnel;

	public:

		__device__ MicrofacetReflection(const Spectrum& R,
			float alphaX, float alphaY, Fresnel* fresnel) :
			BxDF(BxDFType(BSDF_REFLECTION | BSDF_GLOSSY)), R(R), distribution(alphaX, alphaY), fresnel(fresnel) {}

		__device__ void clean() override {
			free(fresnel);
		}

		__device__ Spectrum f(const Vector3f& wo, const Vector3f& wi) const override {
			float cosThetaO = absCosTheta(wo), cosThetaI = absCosTheta(wi);
			Vector3f wh = wi + wo;
			if (cosThetaI == 0 || cosThetaO == 0) return Spectrum(0.0f);
			if (wh.x == 0.0f && wh.y == 0.0f && wh.z == 0.0f) return Spectrum(0.0f);

			wh = normalize(wh);
			Spectrum f = fresnel->evaluate(dot(wi, wh));
			return R * distribution.d(wh) * distribution.g(wo, wi) * f / (4 * cosThetaI * cosThetaO);
		}

		__device__ Spectrum sampleF(const Vector3f& wo, Vector3f* wi,
			Point2f& sample, float* pdf, BxDFType* sampledType = nullptr) const override {

			if (wo.z == 0) return 0.0f;
			Vector3f wh = distribution.sampleWH(wo, sample);
			if (dot(wo, wh) < 0) return 0.0f;
			*wi = reflect(wo, wh);
			if (!sameHemisphere(wo, *wi)) return Spectrum(0.0f);

			*pdf = distribution.Pdf(wo, wh) / (4 * dot(wo, wh));
			return f(wo, *wi);
		}

		__device__ float Pdf(const Vector3f& wo, const Vector3f& wi) const override {
			if (!sameHemisphere(wo, wi)) return 0;
			Vector3f wh = normalize(wo + wi);
			return distribution.Pdf(wo, wh) / (4 * dot(wo, wh));
		}
	};

	// microfacet transmission bxdf
	class MicrofacetTransmission : public BxDF {

		const Spectrum T;
		const TrowbridgeReitzMicrofacetDistribution distribution;
		const float etaA, etaB;
		const FresnelDielectric fresnel;

	public:

		__device__ MicrofacetTransmission(const Spectrum& T,
			float alphaX, float alphaY, float etaA, float etaB) :
			BxDF(BxDFType(BSDF_TRANSMISSION | BSDF_GLOSSY)), T(T), distribution(alphaX, alphaY),
			etaA(etaA), etaB(etaB),
			fresnel(etaA, etaB) {}

		__device__ Spectrum f(const Vector3f& wo, const Vector3f& wi) const override {
			if (sameHemisphere(wo, wi)) return 0.0f;

			float cosThetaO = cosTheta(wo);
			float cosThetaI = cosTheta(wi);
			if (cosThetaI == 0 || cosThetaO == 0) return Spectrum(0.0f);

			float eta = cosTheta(wo) > 0 ? (etaB / etaA) : (etaA / etaB);
			Vector3f wh = normalize(wo + wi * eta);
			if (wh.z < 0) wh = -wh;

			if (dot(wo, wh) * dot(wi, wh) > 0) return Spectrum(0.0f);

			Spectrum f = fresnel.evaluate(dot(wo, wh));

			float sqrtDenom = dot(wo, wh) + eta * dot(wi, wh);

			return (Spectrum(1.0f) - f) * T *
				fabsf(distribution.d(wh) * distribution.g(wo, wi) * eta * eta *
					absDot(wi, wh) * absDot(wo, wh) /
					(cosThetaI * cosThetaO * sqrtDenom * sqrtDenom));
		}

		__device__ Spectrum sampleF(const Vector3f& wo, Vector3f* wi,
			Point2f& sample, float* pdf, BxDFType* sampledType = nullptr) const override {

			if (wo.z == 0) return 0.0f;
			Vector3f wh = distribution.sampleWH(wo, sample);
			if (dot(wo, wh) < 0) return 0.0f;

			float eta = cosTheta(wo) > 0 ? (etaA / etaB) : (etaB / etaA);
			if (!refract(wo, (Normal3f)wh, eta, wi)) return 0.0f;
			*pdf = Pdf(wo, *wi);
			return f(wo, *wi);
		}

		__device__ float Pdf(const Vector3f& wo, const Vector3f& wi) const override {
			if (sameHemisphere(wo, wi)) return 0;

			float eta = cosTheta(wo) > 0 ? (etaA / etaB) : (etaB / etaA);
			Vector3f wh = normalize(wo + wi * eta);

			if (dot(wo, wh) * dot(wi, wh) > 0) return 0.0f;

			float sqrtDenom = dot(wo, wh) + eta * dot(wi, wh);
			float dwh_dwi = fabsf((eta * eta * dot(wi, wh)) / (sqrtDenom * sqrtDenom));
			return distribution.Pdf(wo, wh) * dwh_dwi;
		}
	};
}

#endif