#ifndef __material_cuh__
#define __material_cuh__

#include "BSDF.cuh"
#include "models/LambertianReflection.cuh"
#include "models/Specular.cuh"
#include "models/Microfacet.cuh"
#include "models/Blend.cuh"
#include "textures/Texture.cuh"

namespace lts {

	class Material {

	public:
		// add arena
		__device__ virtual void computeScatteringFunctions(SurfaceInteraction* si, TransportMode mode) const {}
		__device__ virtual ~Material() {}
		// bump
	};

	class MatteMaterial : public Material {

	public:
		Texture<Spectrum>* Kd;
		float roughness;

		__device__ MatteMaterial(Texture<Spectrum>* Kd, float roughness) : Kd(Kd), roughness(roughness) {}

		__device__ void computeScatteringFunctions(SurfaceInteraction* si, TransportMode mode) const override {

			si->bsdf = new BSDF(*si);
			assert(si->bsdf);

			Spectrum kd = Kd->evaluate(*si);

			if (!kd.isBlack()) {
				if (roughness == 0) {
					si->bsdf->add(new LambertianReflection(kd));
				}
				else {
					si->bsdf->add(new OrenNayar(kd, roughness));
				}
			}
		}
	};

	class GlassMaterial : public Material {

		const float eta;
		Texture<Spectrum>* R, * T;

	public:

		__device__ GlassMaterial(float eta, Texture<Spectrum>* R, Texture<Spectrum>* T) : eta(eta), R(R), T(T) {}

		__device__ void computeScatteringFunctions(SurfaceInteraction* si, TransportMode mode) const override {

			si->bsdf = new BSDF(*si, eta);
			assert(si->bsdf);

			Spectrum r = R->evaluate(*si), t = T->evaluate(*si);
			if (r.isBlack() && t.isBlack()) return;

			si->bsdf->add(new FresnelSpecular(r, t, 1.0f, eta, mode));
		}
	};

	class MirrorMaterial : public Material {

		Texture<Spectrum>* R;

	public:

		__device__ MirrorMaterial(Texture<Spectrum>* R) : R(R) {}

		__device__ void computeScatteringFunctions(SurfaceInteraction* si, TransportMode mode) const override {

			si->bsdf = new BSDF(*si);
			assert(si->bsdf);

			Spectrum r = R->evaluate(*si);
			if (r.isBlack()) return;
			si->bsdf->add(new SpecularReflection(r, new FresnelNoOp));
		}
	};

	class MetalMaterial : public Material {

		Texture<Spectrum>* K;
		Spectrum eta;
		float roughness;

	public:

		__device__ MetalMaterial(Texture<Spectrum>* K, Spectrum eta, float roughness) : K(K), eta(eta),
			roughness(roughness) {}

		__device__ void computeScatteringFunctions(SurfaceInteraction* si, TransportMode mode) const override {

			si->bsdf = new BSDF(*si);
			assert(si->bsdf);

			Spectrum k = K->evaluate(*si);
			Fresnel* frMf = new FresnelConductor(1.0f, eta, k);

			float uRough = TrowbridgeReitzMicrofacetDistribution::roughnessToAlpha(roughness);
			float vRough = TrowbridgeReitzMicrofacetDistribution::roughnessToAlpha(roughness);

			si->bsdf->add(new MicrofacetReflection(Spectrum(1.0f), uRough, vRough, frMf));
		}
	};
}

#endif