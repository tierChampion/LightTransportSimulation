#ifndef __lambertian_refl_cuh__
#define __lambertian_refl_cuh__

#include "BxDF.cuh"

namespace lts {

	class LambertianReflection : public BxDF {

		const Spectrum R;

	public:

		__device__ LambertianReflection(const Spectrum& R) : BxDF(BxDFType(BSDF_REFLECTION | BSDF_DIFFUSE)), R(R) {}

		__device__ Spectrum f(const Vector3f& wo, const Vector3f& wi) const override {
			return R * M_1_PI;
		}

		__device__ Spectrum rho(const Vector3f& wo, int nSamples,
			const Point2f* samples) const override {
			return R;
		}
		__device__ Spectrum rho(int nSamples,
			const Point2f* samples1, const Point2f* samples2) const override {
			return R;
		}

	};

}

#endif