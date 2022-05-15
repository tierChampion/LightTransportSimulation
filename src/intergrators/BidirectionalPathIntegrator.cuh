#ifndef __bidirectional_cuh__
#define __bidirectional_cuh__

#include "IntegratorHelper.cuh"

namespace lts {

	// to debug, not functional for now...

	__device__ inline float correctShadingNormal(const SurfaceInteraction& si, const Vector3f& wo,
		const Vector3f& wi, TransportMode mode) {
		if (mode == TransportMode::Importance) {

			float denom = absDot(wo, si.it.n) * absDot(wi, si.shading.n);
			if (denom == 0) return 0.0f;

			return (absDot(wo, si.shading.n) * absDot(wi, si.it.n)) / denom;
		}
		else {
			return 1.0f;
		}

	}

	enum class VertexType {
		Camera,
		Light,
		Surface,
	};

	struct EndpointInteraction {

		Interaction it;

		union {
			const Camera* camera;
			const Light* light;
		};

		__device__ EndpointInteraction() : it(), light(nullptr) {}
		__device__ EndpointInteraction(const Interaction& it, const Camera* camera) :
			it(it), camera(camera) {}
		__device__ EndpointInteraction(const Camera* camera, const Ray& ray) :
			camera(camera), it(ray.o) {}
		__device__ EndpointInteraction(const Interaction& it, const Light* light) :
			it(it), light(light) {}
		__device__ EndpointInteraction(const Light* light, const Ray& ray, const Normal3f nl) :
			light(light), it(ray.o) {
			it.n = nl;
		}
		__device__ EndpointInteraction(const Ray& ray) :
			it(ray.o), light(nullptr) {
			it.n = Normal3f(-ray.d);
		}
	};

	template <typename T>
	class ScopedAssignement {

		T* target, backup;

	public:

		__device__ ScopedAssignement(T* target = nullptr, T value = T()) : target(target) {
			if (target) {
				backup = *target;
				*target = value;
			}
		}

		__device__ ~ScopedAssignement() { if (target) *target = backup; }

		__device__ ScopedAssignement& operator=(ScopedAssignement&& other) {
			target = other.target;
			backup = other.backup;
			other.target = nullptr;
			return *this;
		}

	};

	struct Vertex {

		VertexType type;
		Spectrum beta;
		union {
			SurfaceInteraction si;
			EndpointInteraction ei;
		};

		bool delta = false;
		float pdfFwd, pdfRev;

		__device__ Vertex() : ei() {}
		__device__ Vertex(VertexType type, const EndpointInteraction& ei,
			const Spectrum& beta) :
			type(type), beta(beta), ei(ei) {}
		__device__ Vertex(const SurfaceInteraction& si, const Spectrum& beta) :
			type(VertexType::Surface), beta(beta), si(si) {}

		__device__ ~Vertex() {}

		__device__ const Interaction& getInteraction() const {
			if (type == VertexType::Surface) return si.it;
			else return ei.it;
		}

		__device__ const Point3f& p() const {
			return getInteraction().p;
		}

		__device__ const Normal3f& ng() const {
			return getInteraction().n;
		}

		__device__ const Normal3f& ns() const {
			if (type == VertexType::Surface) return si.shading.n;
			else return getInteraction().n;
		}

		__device__ bool isOnSurface() const {
			return ng() != Normal3f();
		}

		__device__ Spectrum f(const Vertex& next, TransportMode mode) const {
			Vector3f wi = next.p() - p();
			if (wi.lengthSquared() == 0.0f) return 0.0f;
			wi = normalize(wi);
			if (type == VertexType::Surface)
				return si.bsdf->f(si.it.wo, wi) * correctShadingNormal(si, si.it.wo, wi, mode);
		}

		__device__ bool isConnectible() const {
			return !(type == VertexType::Surface && si.bsdf->numComponents(
				BxDFType(BSDF_DIFFUSE | BSDF_GLOSSY | BSDF_REFLECTION | BSDF_TRANSMISSION)) == 0) &&
				!(type == VertexType::Light && (ei.light->flag & (int)LightFlags::DeltaDirection == 1));
		}

		__device__ bool isLight() const {
			return type == VertexType::Light ||
				(type == VertexType::Surface && si.primitive->getAreaLight());
		}

		__device__ bool isDeltaLight() const {
			return type == VertexType::Light && ei.light && lts::isDeltaLight(ei.light->flag);
		}

		__device__ Spectrum Le(const Scene& scene, const Vertex& v) const {

			if (!isLight()) return 0.0f;
			Vector3f w = v.p() - p();
			if (w.lengthSquared() == 0.0f) return 0.0f;
			w = normalize(w);

			const AreaLight* light = si.primitive->getAreaLight();
			assert(light);
			return light->L(si.it, w);
		}

		__device__ float convertDensity(float pdf, const Vertex& next) const {
			Vector3f w = next.p() - p();
			if (w.lengthSquared() == 0.0f) return 0.0f;
			float invDist2 = 1 / w.lengthSquared();
			if (next.isOnSurface()) pdf *= absDot(next.ng(), w * sqrtf(invDist2));
			return pdf * invDist2;
		}

		__device__ float PdfLightOrigin(const Scene& scene, const Vertex& v,
			const Distribution1D& lightDistrib) const {

			Vector3f w = v.p() - p();
			if (w.lengthSquared() == 0.0f) return 0.0f;
			w = normalize(w);

			float pdfChoice, pdfPos, pdfDir;

			const Light* light = type == VertexType::Light ? ei.light : si.primitive->getAreaLight();

			for (size_t i = 0; i < scene.lightCount; i++) {
				if (scene.lights[i] == light) {
					pdfChoice = lightDistrib.discretePdf(i);
					break;
				}
			}

			light->PdfLe(Ray(p(), w), ng(), &pdfPos, &pdfDir);
			return pdfChoice * pdfPos;
		}

		__device__ float PdfLight(const Scene& scene, const Vertex& v) const {

			Vector3f w = v.p() - p();
			float invDist2 = 1 / w.lengthSquared();
			w *= sqrtf(invDist2);
			float pdf;

			const Light* light = type == VertexType::Light ? ei.light : si.primitive->getAreaLight();
			float pdfPos, pdfDir;
			light->PdfLe(Ray(p(), w), ng(), &pdfPos, &pdfDir);
			pdf = pdfDir * invDist2;

			if (v.isOnSurface()) pdf *= absDot(v.ng(), w);
			return pdf;
		}

		__device__ float Pdf(const Scene& scene, const Vertex* prev, const Vertex& next) const {
			// handle light
			if (type == VertexType::Light) return PdfLight(scene, next);
			//compute directions
			Vector3f wp, wn = next.p() - p();
			if (wn.lengthSquared() == 0.0f) return 0.0f;
			wn = normalize(wn);
			if (prev) {
				wp = prev->p() - p();
				if (wp.lengthSquared() == 0.0f) return 0.0f;
				wp = normalize(wp);
			}
			//compute directionnal density
			float pdf, unused;
			if (type == VertexType::Camera) ei.camera->PdfWe(ei.it.spawnRay(wn), &unused, &pdf);
			else if (type == VertexType::Surface) si.bsdf->Pdf(wp, wn);
			//return probability
			return convertDensity(pdf, next);
		}

		// todo - very simple
		__device__ static inline Vertex createCamera(const Camera* camera, const Ray& ray,
			const Spectrum& beta);
		__device__ static inline Vertex createCamera(const Camera* camera, const Interaction& it,
			const Spectrum& beta);
		__device__ static inline Vertex createLight(const Light* light, const Ray& ray,
			const Normal3f nLight, const Spectrum& Le, float pdf);
		__device__ static inline Vertex createLight(const EndpointInteraction& ei,
			const Spectrum& beta, float pdf);
		__device__ static inline Vertex createSurface(const SurfaceInteraction& si,
			const Spectrum& beta, float pdf, const Vertex& prev);
	};

	__device__ inline Vertex Vertex::createCamera(const Camera* camera, const Ray& ray,
		const Spectrum& beta) {

		return Vertex(VertexType::Camera, EndpointInteraction(camera, ray), beta);
	}

	__device__ inline Vertex Vertex::createCamera(const Camera* camera, const Interaction& it,
		const Spectrum& beta) {
		return Vertex(VertexType::Camera, EndpointInteraction(it, camera), beta);
	}

	__device__ inline Vertex Vertex::createLight(const Light* light, const Ray& ray,
		const Normal3f nLight, const Spectrum& Le, float pdf) {
		Vertex v(VertexType::Light, EndpointInteraction(light, ray, nLight), Le);
		v.pdfFwd = pdf;
		return v;
	}

	__device__ inline Vertex Vertex::createLight(const EndpointInteraction& ei,
		const Spectrum& beta, float pdf) {
		Vertex v(VertexType::Light, ei, beta);
		v.pdfFwd = pdf;
		return v;
	}

	__device__ inline Vertex Vertex::createSurface(const SurfaceInteraction& si,
		const Spectrum& beta, float pdf, const Vertex& prev) {

		Vertex v(si, beta);
		v.pdfFwd = prev.convertDensity(pdf, v);
		return v;
	}

	/**
	* Does a random walk of maximum length maxDepth starting with a given ray.
	* @param scene - scene to random walk in
	* @param ray - first step in the walk
	* @param sampler - sampler of random variables
	* @param beta - initial radiance
	* @param pdf - initial pdf
	* @param maxDepth - maximum length of the walk
	* @param mode - direction of the walk
	* @param path - pointer to record the walk
	* @param id - pixel being evaluated
	* @return length of the walk
	*/
	__device__ inline int randomWalk(const Scene& scene, Ray ray, Sampler& sampler,
		Spectrum beta, float pdf, int maxDepth, TransportMode mode, Vertex* path, int id) {

		if (maxDepth == 0) return 0;
		int bounces = 0;
		float pdfFwd = pdf, pdfRev = 0;

		while (true) {

			SurfaceInteraction si;
			bool foundIntersect = scene.intersect(ray, &si);
			if (beta.isBlack()) break;
			Vertex& vertex = path[bounces], & prev = path[bounces - 1];

			if (!foundIntersect) {
				if (mode == TransportMode::Radiance) {
					vertex = Vertex::createLight(EndpointInteraction(ray), beta, pdfFwd);
					bounces++;
				}
				break;
			}

			si.computeScatteringFunctions(ray, mode);

			vertex = Vertex::createSurface(si, beta, pdfFwd, prev);

			if (++bounces >= maxDepth) break;

			Vector3f wi, wo = si.it.wo;
			BxDFType type;
			Spectrum f = si.bsdf->sampleF(wo, &wi, sampler.uniformSample(id), &pdfFwd, BSDF_ALL, &type);

			if (pdfFwd == 0 || f.isBlack()) break;

			beta *= f * absDot(wi, si.shading.n) / pdfFwd;
			pdfRev = si.bsdf->Pdf(wi, wo, BSDF_ALL);

			if (type & BSDF_SPECULAR) {
				vertex.delta = true;
				pdfFwd = pdfRev = 0.0f;
			}

			beta *= correctShadingNormal(si, wo, wi, mode);

			ray = si.it.spawnRay(wi);

			prev.pdfRev = vertex.convertDensity(pdfRev, prev);
		}

		return bounces;
	}

	/**
	* Generate a subpath starting from the camera
	* @param scene - scene to generate the path in
	* @param sampler - sampler of random variables
	* @param maxDepth - maximum length of the subpath
	* @param camera - camera to start the path at
	* @param pFilm - location on the film to sample
	* @param path - pointer to record the path
	* @param id - pixel being evaluated
	* @return length of the camera subpath
	*/
	__device__ inline int generateCameraSubpath(const Scene& scene, Sampler& sampler,
		int maxDepth, const Camera& camera, const Point2f& pFilm, Vertex* path, int id) {

		if (maxDepth == 0) return 0;

		Point2f pLens = sampler.uniformSample(id);

		Ray ray = camera.generateDifferentialRay(pFilm, pLens);
		ray.scaleDifferentials(quakeIIIFastInvSqrt(sampler.samplesPerPixel));
		Spectrum beta = 1.0f;

		float pdfPos, pdfDir;
		path[0] = Vertex::createCamera(&camera, ray, beta);
		camera.PdfWe(ray, &pdfPos, &pdfDir);
		return randomWalk(scene, ray, sampler, beta, pdfDir, maxDepth - 1,
			TransportMode::Radiance, path + 1, id) + 1;
	}

	/**
	* Generate a subpath starting from a random light in the scene
	* @param scene - scene to generate the path in
	* @param sampler - sampler of random variables
	* @param maxDepth - maximum length of the subpath
	* @param lightDistrib - distribution of the power of the lights in the scene
	* @param path - pointer to record the path
	* @param id - pixel being evaluated
	* @return length of the light subpath
	*/
	__device__ inline int generateLightSubpath(const Scene& scene, Sampler& sampler,
		int maxDepth, const Distribution1D& lightDistrib, Vertex* path, int id) {

		if (maxDepth == 0) return 0;

		float lightPdf;
		int lightNum = lightDistrib.sampleDiscrete(sampler.get1DSample(id), &lightPdf);
		const Light* light = scene.lights[lightNum];

		Ray ray;
		Normal3f nLight;
		float pdfPos, pdfDir;
		Spectrum Le = light->sampleLe(sampler.uniformSample(id), sampler.uniformSample(id),
			&ray, &nLight, &pdfPos, &pdfDir);
		if (pdfPos == 0 || pdfDir == 0 || Le.isBlack()) return 0;

		path[0] = Vertex::createLight(light, ray, nLight, Le, pdfPos * lightPdf);
		Spectrum beta = Le * absDot(nLight, ray.d) / (lightPdf * pdfPos * pdfDir);
		int nVertices = randomWalk(scene, ray, sampler, beta, pdfDir, maxDepth - 1,
			TransportMode::Importance, path, id);

		return nVertices + 1;
	}

	/**
	* Computes the generalized geometric term of the LTE
	* @param scene - scene to evaluate
	* @param v0 and v1 - vertices
	*/
	__device__ inline Spectrum g(const Scene& scene, const Vertex& v0, const Vertex& v1) {

		Vector3f d = v0.p() - v1.p();
		float g = 1 / d.lengthSquared();
		d *= sqrtf(g);
		if (v0.isOnSurface()) {
			g *= absDot(v0.ns(), d);
		}
		if (v1.isOnSurface()) {
			g *= absDot(v1.ns(), d);
		}

		VisibilityTester vis(v0.getInteraction(), v1.getInteraction());
		return (vis.unoccluded(scene) ? Spectrum(g) : Spectrum(0.0f));
	}

	/**
	* Computes the multiple importance sampling weight of the given path strategy
	* @param scene - scene to evaluate
	* @param lightPath - section of the total path starting from the light
	* @param cameraPath - section of the total path starting from the camera
	* @param sampled - resampled origin points, if needed
	* @param s - length of light path
	* @param t - length of camera path
	* @param lightDistrib - distribution of the lights in the scene
	*/
	__device__ inline float MISWeight(const Scene& scene, Vertex* lightPath, Vertex* cameraPath,
		Vertex sampled, int s, int t, const Distribution1D& lightDistrib) {

		// Only a single possible path composition
		if (s + t == 2) return 1;

		float sumRi = 0;

		auto remap0 = [](float f) -> float {return f != 0 ? f : 1; };

		Vertex* qs = s > 0 ? &lightPath[s - 1] : nullptr,
			* pt = t > 0 ? &cameraPath[t - 1] : nullptr,
			* qsMinus = s > 1 ? &lightPath[s - 2] : nullptr,
			* ptMinus = t > 1 ? &cameraPath[t - 2] : nullptr;

		ScopedAssignement<Vertex> a1;
		if (s == 1) a1 = { qs, sampled };
		else if (t == 1) a1 = { pt, sampled };

		ScopedAssignement<bool> a2, a3;
		if (pt) a2 = { &pt->delta, false };
		if (qs) a3 = { &qs->delta, false };

		ScopedAssignement<float> a4;
		if (pt) {
			a4 = { &pt->pdfRev,
			s > 0 ? qs->Pdf(scene, qsMinus, *pt) :
			pt->PdfLightOrigin(scene, *ptMinus, lightDistrib) };
		}

		ScopedAssignement<float> a5;
		if (ptMinus) {
			a5 = { &ptMinus->pdfRev,
				s > 0 ? pt->Pdf(scene, qs, *ptMinus) :
				pt->PdfLight(scene, *ptMinus) };
		}

		ScopedAssignement<float> a6;
		if (qs) a6 = { &qs->pdfRev, pt->Pdf(scene, ptMinus, *qs) };
		ScopedAssignement<float> a7;
		if (qsMinus) a7 = { &qsMinus->pdfRev, qs->Pdf(scene, pt, *qsMinus) };

		float ri = 1;
		for (int i = t - 1; i > 0; i--) {
			ri *= remap0(cameraPath[i].pdfRev) / remap0(cameraPath[i].pdfFwd);
			if (!cameraPath[i].delta && !cameraPath[i - 1].delta)
				sumRi += ri;
		}

		ri = 1;
		for (int i = s - 1; i >= 0; i--) {
			ri *= remap0(lightPath[i].pdfRev) / remap0(lightPath[i].pdfFwd);
			bool deltaLightVertex = i > 0 ? lightPath[i - 1].delta :
				lightPath[0].isDeltaLight();
			if (!lightPath[i].delta && !deltaLightVertex) sumRi += ri;
		}

		return 1 / (1 + sumRi);
	}

	/**
	* Calculate the radiance contribution of the connected paths.
	* @param scene - scene to evaluate
	* @param lightPath - path starting at the light
	* @param cameraPath - path starting at the camera
	* @param s - length of the light path
	* @param t - length of the camera path
	* @param camera - camera at the origin of the camera path
	* @param lightDistrib - distribution of the lights in the scene
	* @param sampler - sampler of random variables
	* @param id - pixel being evaluated
	* @param pRaster - point on the raster of the final path. Optional
	* @param misWeightPtr - weight of the multiple importance sampling. Optional
	*/
	__device__ inline Spectrum connectBDPT(const Scene& scene, Vertex* lightPath, Vertex* cameraPath,
		int s, int t, const Camera& camera, const Distribution1D& lightDistrib,
		Sampler& sampler, int id, Point2f* pRaster, float* misWeightPtr) {

		Spectrum L(0.0f);
		if (t > 1 && s != 0 && cameraPath[t - 1].type == VertexType::Light) return Spectrum(0.0f);

		Vertex sampled;

		if (s == 0) {
			const Vertex& pt = cameraPath[t - 1];
			if (pt.isLight()) L = pt.Le(scene, cameraPath[t - 2]) * pt.beta;
		}
		else if (t == 1) {
			const Vertex& qs = lightPath[s - 1];
			if (qs.isConnectible()) {
				VisibilityTester vis;
				Vector3f wi;
				float pdf;
				Spectrum Wi = camera.sampleWi(qs.getInteraction(), sampler.uniformSample(id),
					&wi, &pdf, pRaster, &vis);

				if (pdf > 0 && !Wi.isBlack()) {
					sampled = Vertex::createCamera(&camera, vis.p1, Wi / pdf);
					L = qs.beta * qs.f(sampled, TransportMode::Importance) * sampled.beta;
					if (qs.isOnSurface()) L *= absDot(wi, qs.ns());
					if (!L.isBlack()) L = vis.unoccluded(scene) ? L : Spectrum(0.0f);
				}
			}
		}
		else if (s == 1) {

			const Vertex& pt = cameraPath[t - 1];
			if (pt.isConnectible()) {
				float lightPdf;
				VisibilityTester vis;
				Vector3f wi;
				float pdf;
				int lightNum = lightDistrib.sampleDiscrete(sampler.get1DSample(id), &lightPdf);
				const Light* light = scene.lights[lightNum];
				Spectrum lightWeight = light->sampleLi(pt.getInteraction(), sampler.uniformSample(id),
					&wi, &pdf, &vis);

				if (pdf > 0 && !lightWeight.isBlack()) {
					EndpointInteraction ei(vis.p1, light);
					sampled = Vertex::createLight(ei, lightWeight / (pdf * lightPdf), 0);
					sampled.pdfFwd = sampled.PdfLightOrigin(scene, pt, lightDistrib);
					L = pt.beta * pt.f(sampled, TransportMode::Radiance) * sampled.beta;
					if (pt.isOnSurface()) {
						L *= absDot(wi, pt.ns());
					}
					if (!L.isBlack()) L = vis.unoccluded(scene) ? L : Spectrum(0.0f);
				}


			}
		}
		else {
			const Vertex& qs = lightPath[s - 1], & pt = cameraPath[t - 1];
			if (qs.isConnectible() && pt.isConnectible()) {

				Spectrum qsf = qs.f(pt, TransportMode::Importance), ptf = pt.f(qs, TransportMode::Radiance);
				L = qs.beta * qsf * ptf * pt.beta;

				//printf("qsbeta = ? ? ?, qsf = %f %f %f, ptf = %f %f %f, ptbeta = ? ? ?\n",
				//	qsf.r, qsf.g, qsf.b, ptf.r, ptf.g, ptf.b);

				if (!L.isBlack()) L *= g(scene, qs, pt);
			}
		}

		float misWeight = L.isBlack() ? 0.0f :
			MISWeight(scene, lightPath, cameraPath, sampled, s, t, lightDistrib);

		L *= misWeight;
		if (misWeightPtr) *misWeightPtr = misWeight;

		return L;
	}

	class BidirectionalPathIntegrator {

		Camera* h_camera, * d_camera;
		Sampler* sampler;

		const int MAX_BOUNCE, ROULETTE_START;

	public:

		Scene* scene;

		__host__ BidirectionalPathIntegrator(int maxDepth, int rrStart,
			Camera* h_cam, Sampler* d_samp, Scene* d_scene) :
			MAX_BOUNCE(maxDepth), ROULETTE_START(rrStart),
			h_camera(h_cam), sampler(d_samp), scene(d_scene)
		{
			d_camera = passToDevice(h_cam);
		}

		__device__ void evaluate(int i, int j, int seed, const Distribution1D& distribution) {

			// 1. Check if pixel is valid
			Vector2i dimensions = d_camera->filmResolution.diagonal();

			if (i >= dimensions.x || j >= dimensions.y) return;

			int id = j * dimensions.x + i;

			// 2. Prepare sampler
			sampler->prepareThread(id, seed);

			for (int s = 0; s < sampler->samplesPerPixel; s++) {

				Point2f filmSample = sampler->uniformSample(id);
				Point2f pFilm = Point2f((float)(i + filmSample.x),
					(float)(j + filmSample.y));

				// allocate vertices
				Vertex* cameraPath = new Vertex[MAX_BOUNCE + 2];
				Vertex* lightPath = new Vertex[MAX_BOUNCE + 1];
				// create paths
				int nCamera = generateCameraSubpath(*scene, *sampler, MAX_BOUNCE + 2,
					*d_camera, pFilm, cameraPath, id);
				int nLight = generateLightSubpath(*scene, *sampler, MAX_BOUNCE + 1,
					distribution, lightPath, id);


				Spectrum L(0.0f);
				for (int t = 1; t < nCamera; t++) {
					for (int s = 0; s < nLight; s++) {
						int depth = t + s - 2;
						if ((s == 1 && t == 1) || depth < 0 || depth > MAX_BOUNCE) continue;
						Point2f pFilmNew = pFilm;
						float misWeight = 0.0f;
						Spectrum Lpath = connectBDPT(*scene, lightPath, cameraPath, s, t,
							*d_camera, distribution, *sampler, id, &pFilmNew, &misWeight);
						if (t != 1) L += Lpath;
						else d_camera->addSplat(pFilmNew, Lpath);
					}
				}

				d_camera->addSample(pFilm, L);

				free(cameraPath);
				free(lightPath);
			}
		}

		__host__ void outputResultAndOpen(std::string outputFile, int format) {

			h_camera->sendRenderToImageFile(outputFile, format, 1.0f);

			// open render
			system(outputFile.c_str());
		}
	};

	__global__ void BidirectionalPathTracingKernel(BidirectionalPathIntegrator* BDPT_integrator,
		Distribution1D* l_distrib, unsigned int seed);
}

#endif