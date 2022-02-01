#ifndef __integrator_helper_cuh__
#define __integrator_helper_cuh__

#include "../rendering/Camera.cuh"
#include "../acceleration/Scene.cuh"
#include "../sampling/Sampler.cuh"

namespace lts {

	enum IntegratorType {
		PathTracing,
		BiderectionalPathTracing,
		Metropolis
	};

	__host__ inline const char* toString(IntegratorType type) {
		switch (type) {
		case PathTracing: return "Path Tracing";
		case BiderectionalPathTracing: return "Bidirectional Path Tracing";
		case Metropolis: return "Metropolis Light Transport";
		default: return "No set algorithms!";
		}
	}

	__device__ inline Distribution1D computeLightPowerDistribution(const Scene& scene) {

		if (scene.lightCount == 0) return Distribution1D(nullptr, 0);
		float* lightPower = new float[scene.lightCount];
		for (int l = 0; l < scene.lightCount; l++) {
			lightPower[l] = scene.lights[l]->power().y();
		}

		return Distribution1D(lightPower, scene.lightCount);
	}

	/**
	* Estimates the direct lighting component of the integral at location it for a given light.
	* @param it - location in the scene
	* @param scatteringSample - random sample for the scattering
	* @param light - light to estimate the direct lighting from
	* @param lightSample - random sample for the sampling of the light
	* @param scene - scene to intergrate
	* @param specular - if the bounce is specular.
	*/
	__device__ inline Spectrum estimateDirect(const SurfaceInteraction& si, const Point2f& scatteringSample, const Light& light,
		const Point2f& lightSample, const Scene& scene, bool specular) {

		Spectrum Ld(0.0f);
		BxDFType flags = specular ? BSDF_ALL : BxDFType(BSDF_ALL & ~BSDF_SPECULAR);

		Vector3f wi;
		float lightPdf = 0.0f, scatteringPdf = 0.0f;
		VisibilityTester visibility;
		Spectrum Li = light.sampleLi(si.it, lightSample, &wi, &lightPdf, &visibility);

		if (lightPdf > 0 && !Li.isBlack()) {
			Spectrum f;

			f = si.bsdf->f(si.it.wo, wi, flags);
			scatteringPdf = si.bsdf->Pdf(si.it.wo, wi, flags);

			if (!f.isBlack()) {
				if (!visibility.unoccluded(scene)) {
					Li = Spectrum(0.0f);
				}

				if (!Li.isBlack()) {
					if (isDeltaLight(light.flag)) {
						Ld += f * Li / lightPdf;
					}
					else {
						float weight = powerHeuristic(1, lightPdf, 1, scatteringPdf);
						Ld += f * Li * weight / lightPdf;
					}
				}
			}
		}

		if (!isDeltaLight(light.flag)) {
			Spectrum f;
			bool sampledSpecular = false;

			BxDFType sampledType;
			f = si.bsdf->sampleF(si.it.wo, &wi, scatteringSample, &scatteringPdf, flags, &sampledType);
			f *= absDot(wi, si.shading.n);
			sampledSpecular = sampledType & BSDF_SPECULAR;

			if (!f.isBlack() && scatteringPdf > 0) {

				float weight = 1;
				if (!sampledSpecular) {
					lightPdf = light.PdfLi(si.it, wi);
					if (lightPdf == 0) return Ld;
					weight = powerHeuristic(1, scatteringPdf, 1, lightPdf);
				}

				SurfaceInteraction lightIt;
				Ray r = si.it.spawnRay(wi);
				Spectrum Tr(1.0f);

				bool foundSurfaceInteraction = scene.intersect(r, &lightIt);

				Spectrum Li(0.f);
				if (foundSurfaceInteraction) {
					if (lightIt.primitive->getAreaLight() == &light) {
						Li = lightIt.Le(-wi);
					}
				}
				if (!Li.isBlack()) {
					Ld += f * Li * Tr * weight / scatteringPdf;
				}
			}
		}
		return Ld;
	}

	/**
	* Sample the direct lighting of one random light in the scene.
	* @param it - location in the scene
	* @param scene - scene to integrate
	* @param sampler - Random sample generator
	* @param specular - if the bounce is specular
	* @param id - index of the current pixel being estimated
	*/
	__device__ inline Spectrum uniformSampleOneLight(const SurfaceInteraction& it, const Scene& scene,
		const Sampler& sampler, const Distribution1D& lightDistrib,
		bool specular, int id) {

		int nLights = scene.lightCount;
		if (nLights == 0) return Spectrum(0.0f);
		float lightPdf;
		int lightNum = lightDistrib.sampleDiscrete(sampler.get1DSample(id), &lightPdf);
		const Light* light = scene.lights[lightNum];

		Point2f lightSample = sampler.uniformSample(id);
		Point2f scatteringSample = sampler.uniformSample(id);

		return estimateDirect(it, scatteringSample, *(light), lightSample, scene, specular) / lightPdf;
	}
}

#endif