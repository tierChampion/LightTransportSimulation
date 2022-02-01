#ifndef __pathtracing_cuh__
#define __pathtracing_cuh__

#include "IntegratorHelper.cuh"

namespace lts {

	class PathTracingIntegrator {

		Camera* h_camera, * d_camera;
		Sampler* sampler;
		Scene* scene;

		const int MAX_BOUNCE, ROULETTE_START;

	public:

		__host__ PathTracingIntegrator(int maxRayBounce, int rrStart,
			Camera* h_cam, Sampler* d_samp, Scene* d_scene) :
			MAX_BOUNCE(maxRayBounce), ROULETTE_START(rrStart),
			h_camera(h_cam), sampler(d_samp), scene(d_scene)
		{
			d_camera = passToDevice(h_cam);
		}

		__device__ void evaluate(int i, int j, const Distribution1D& lightDistrib, int seed) {

			// 1. Check if pixel is valid
			Vector2i dimensions = d_camera->filmResolution.diagonal();

			if (i >= dimensions.x || j >= dimensions.y) return;

			int id = j * dimensions.x + i;

			// 2. Prepare sampler
			sampler->prepareThread(id, seed);

			for (int s = 0; s < sampler->samplesPerPixel; s++) {

				// 3. Generate ray for specific sample of the current pixel
				Point2f filmSample = sampler->uniformSample(id);
				Point2f lensSample = uniformSampleDisk(sampler->uniformSample(id));
				Point2f pFilm = Point2f((float)(i + filmSample.x),
					(float)(j + filmSample.y));
				Ray r = d_camera->generateDifferentialRay(pFilm, lensSample);
				r.scaleDifferentials(quakeIIIFastInvSqrt(sampler->samplesPerPixel));
				Spectrum L(0.0f), beta(1.0f);
				bool specularBounce = false;

				// 4. Start ray tracing
				for (int bounce = 0; bounce < MAX_BOUNCE; bounce++) {

					// 5. Intersect with scene and record properties at intersection
					SurfaceInteraction si;
					bool intersectionWithScene = scene->intersect(r, &si);

					// 6. Account for emissive materials (todo)
					if (bounce == 0 || specularBounce) {
						if (intersectionWithScene)
							L += beta * si.Le(-r.d);
						else
							L += beta * Spectrum(0.0f);
					}

					if (!intersectionWithScene) {
						//L += beta * Spectrum(0.5f);
						break;
					}

					// 7. Account for the effects of the BSDF
					si.computeScatteringFunctions(r);

					if (!si.bsdf) break;

					// 7a. Accumulate direct lighting

					L += beta * uniformSampleOneLight(si, *scene, *sampler, lightDistrib, specularBounce, id);

					// 7b. Sample BSDF
					Vector3f wo = -r.d, wi;
					float pdf;
					BxDFType flags;
					Spectrum f = si.bsdf->sampleF(wo, &wi, sampler->uniformSample(id), &pdf, BSDF_ALL, &flags);
					// check if pdf and f are valid
					if (f.isBlack() || pdf == 0) {
						break;
					}
					specularBounce = (flags & BSDF_SPECULAR) != 0;

					// 7c. Update path throughput weight
					beta *= f * absDot(wi, si.shading.n) / pdf;

					// 8. Set new ray
					r = si.it.spawnRay(wi);

					// 9. Terminate paths with low influence to the final color
					if (bounce >= ROULETTE_START) {
						float q = fmaxf(0.05f, 1 - beta.y());
						if (sampler->get1DSample(id) < q) {
							break;
						}
						beta /= 1 - q;
					}
				}

				// 10. Add sample to the film 
				d_camera->addSample(pFilm, L);
			}
		}

		__host__ void outputResultAndOpen(std::string outputFile, int format) {

			h_camera->sendRenderToImageFile(outputFile, format);

			// open render
			system(outputFile.c_str());
		}
	};
}

#endif