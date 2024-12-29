#ifndef __camera_cu__
#define __camera_cu__

#include "Camera.cuh"
#include "../core/ImageO.cuh"

namespace lts {

	__host__ Camera::Camera(const Point3f& lookFrom, const Point3f& lookAt, const Vector3f& vUp,
		float vertFov, float aperture, float focusDist, int imgWidth, int imgHeight, Filter* filter) {

		float aspectRatio = (float)imgWidth / (float)imgHeight;

		float theta = vertFov * M_PI / 180.0f;
		float h = tanf(theta / 2);

		float viewportHeight = 2 * h;
		float viewportWidth = aspectRatio * viewportHeight;

		w = normalize(lookFrom - lookAt);
		u = normalize(cross(vUp, w));
		v = cross(w, u);

		o = lookFrom;
		right = u * (focusDist * viewportWidth);
		up = v * (focusDist * viewportHeight);
		llc = o - (right * 0.5f) - (up * 0.5f) - w * focusDist;

		lensRadius = aperture / 2;
		focalDist = focusDist;

		filmResolution = Bounds2i(Point2i(0, 0), Point2i(imgWidth, imgHeight));

		// Passing the film to the GPU
		gpuErrCheck(cudaMalloc((void**)&d_film, sizeof(Pixel) * filmResolution.area()));

		float* h_table = filter->getFilterTable(FILTER_TABLE_SIZE);
		gpuErrCheck(cudaMalloc((void**)&d_table, sizeof(float) * FILTER_TABLE_SIZE * FILTER_TABLE_SIZE));
		gpuErrCheck(cudaMemcpy(d_table, h_table,
			sizeof(float) * FILTER_TABLE_SIZE * FILTER_TABLE_SIZE, cudaMemcpyHostToDevice));
		filterRadius = filter->radius;
		invFilterRadius = filter->invRadius;
	}

	/**
	* Generate the desired ray from the samples.
	* @param pFilm - Sample on the film
	* @param pLens - Sample on the lens
	* @return Ray passing by both samples
	*/
	__device__ Ray Camera::generateRay(const Point2f& pFilm, const Point2f& pLens) const {

		Vector3f offset = (u * pLens.x + v * pLens.y) * lensRadius;

		Point2f pRaster = Point2f(pFilm.x / (filmResolution.diagonal().x - 1),
			pFilm.y / (filmResolution.diagonal().y - 1));

		return Ray(o + offset, (Vector3f)normalize(llc + right * pRaster.x +
			up * pRaster.y - o - offset));
	}

	/**
	* Generate a ray with differential information on the film.
	* @param pFilm - Sample on the film
	* @param pLens - Sample on the lens
	*/
	__device__ Ray Camera::generateDifferentialRay(const Point2f& pFilm, const Point2f& pLens) const {

		Ray ret = generateRay(pFilm, pLens);

		Ray rx = generateRay(pFilm + Point2f(1, 0), pLens);
		Ray ry = generateRay(pFilm + Point2f(0, 1), pLens);

		ret.rxOrigin = rx.o;
		ret.rxDirection = rx.d;
		ret.ryOrigin = ry.o;
		ret.ryDirection = ry.d;
		ret.hasDifferentials = true;

		return ret;
	}

	/**
	* Evaluates the importance of a given ray
	* @param r - ray to evaluate the importance of
	* @param pRaster2 - location on the raster of the ray. optional
	*/
	__device__ Spectrum Camera::we(const Ray& r, Point2f* pRaster2) const {

		float cosTheta = dot(r.d, -w / focalDist);
		if (cosTheta <= 0) return 0.0f;

		// get point on raster for r
		Point3f pFocus = r(1 / cosTheta);
		Vector3f moveOnRaster = o + (o - r.o) * 2 + pFocus - llc;

		// choose system of equations to avoid nan values
		int e1 = maxDimension(abs(up));
		int e2 = maxDimension(abs(right));
		if (e1 == e2) {
			e2 = e1 < 2 ? e1 + 1 : e1 - 2;
		}

		float denom = 1.0f / (right[e2] * up[e1] - right[e1] * up[e2]);

		float rasterX = (moveOnRaster[e2] * up[e1] * denom) - (moveOnRaster[e1] * up[e2] * denom);
		float rasterY = (-right[e1] * rasterX + moveOnRaster[e1]) / up[e1];

		Point2f pRaster = Point2f(rasterX, rasterY);
		pRaster = pRaster * filmResolution.diagonal();

		if (pRaster2) *pRaster2 = pRaster;

		// Might need to be done later ?
		if (rasterX < 0 || rasterX >= filmResolution.diagonal().x) return 0.0f;
		if (rasterY < 0 || rasterY >= filmResolution.diagonal().y) return 0.0f;

		float lensArea = lensRadius > 0 ? (M_PI * lensRadius * lensRadius) : 1.0f;

		// return importance
		float cos2Theta = cosTheta * cosTheta;
		return Spectrum(1 / (filmResolution.area() * lensArea * cos2Theta * cos2Theta));
	}

	__device__ void Camera::PdfWe(const Ray& r, float* pdfPos, float* pdfDir) const {

		float cosTheta = dot(r.d, -w / focalDist);
		if (cosTheta <= 0) {
			*pdfPos = 0.0f;
			*pdfDir = 0.0f;
			return;
		}

		// get point on raster for r
		Point3f pFocus = r(1 / cosTheta);
		Vector3f moveOnRaster = o + (o - r.o) * 2 + pFocus - llc;

		// choose system of equations to avoid nan values
		int e1 = maxDimension(abs(up));
		int e2 = maxDimension(abs(right));
		if (e1 == e2) {
			e2 = e1 < 2 ? e1 + 1 : e1 - 2;
		}

		float denom = 1.0f / (right[e2] * up[e1] - right[e1] * up[e2]);

		float rasterX = (moveOnRaster[e2] * up[e1] * denom) - (moveOnRaster[e1] * up[e2] * denom);
		float rasterY = (-right[e1] * rasterX + moveOnRaster[e1]) / up[e1];

		// Might need to be done later ?
		if ((rasterX < 0 || rasterX >= 1) ||
			(rasterY < 0 || rasterY >= 1)) {
			*pdfPos = 0.0f;
			*pdfDir = 0.0f;
			return;
		}

		//Point2f pRaster = Point2f(rasterX, rasterY); // Should be from [0, 0] to [1, 1]. Might need to be scaled

		float lensArea = lensRadius > 0 ? (M_PI * lensRadius * lensRadius) : 1.0f;

		*pdfPos = 1 / lensArea;
		*pdfDir = 1 / (filmResolution.area() * cosTheta * cosTheta * cosTheta);
	}

	__device__ Spectrum Camera::sampleWi(const Interaction& it, const Point2f sample,
		Vector3f* wi, float* pdf, Point2f* pRaster, VisibilityTester* vis) const {

		Point2f pLens = uniformSampleDisk(sample) * lensRadius;
		Point3f pLensWorld = o + u * pLens.x + v * pLens.y;

		Interaction lensIntr(pLensWorld);
		lensIntr.n = Normal3f(w);

		*vis = VisibilityTester(it, lensIntr);
		*wi = lensIntr.p - it.p;
		float dist = wi->length();
		*wi /= dist;

		float lensArea = lensRadius > 0 ? (M_PI * lensRadius * lensRadius) : 1.0f;
		// pdf in terms of solid angle
		*pdf = (dist * dist) / (absDot(lensIntr.n, *wi) * lensArea);

		return we(lensIntr.spawnRay(-*wi), pRaster);
	}

	/**
	* Adds a sample on the film with filtering.
	* @param pFilm - location on the film. (range [0, filterWidth])
	* @param L - radiance of the sample
	*/
	__device__ void Camera::addSample(const Point2f& pFilm, const Spectrum L) {

		Point2i p0 = (Point2i)ceil(pFilm - filterRadius);
		Point2i p1 = (Point2i)floor(pFilm + filterRadius) + Point2i(1, 1);

		p0 = max(p0, filmResolution.pMin);
		p1 = min(p1, filmResolution.pMax);

		for (int y = p0.y; y < p1.y; y++) {
			for (int x = p0.x; x < p1.x; x++) {

				// retrieve x and y of the creation of the filter table
				float dx = fabsf((x - pFilm.x - 0.5f) * invFilterRadius.x * FILTER_TABLE_SIZE);
				float dy = fabsf((y - pFilm.y - 0.5f) * invFilterRadius.y * FILTER_TABLE_SIZE);

				int ix = fminf((int)floorf(dx), FILTER_TABLE_SIZE - 1);
				int iy = fminf((int)floorf(dy), FILTER_TABLE_SIZE - 1);

				float filterWeight = d_table[iy * FILTER_TABLE_SIZE + ix];

				int pixel = y * filmResolution.diagonal().x + x;

				d_film[pixel].rgb.atomicAddition(L * filterWeight);
				d_film[pixel].filterWeight += filterWeight;
			}
		}
	}

	/**
	* Adds a splat sample on the film without filtering.
	* @param pFilm - location on the film. (range [0, filterWidth])
	* @param L - radiance of the sample
	*/
	__device__ void Camera::addSplat(const Point2f& pFilm, const Spectrum splat) {

		Point2i p = (Point2i)floor(pFilm);

		if (!insideExclusive(p, filmResolution)) return;

		int pixel = p.y * filmResolution.diagonal().x + p.x;

		d_film[pixel].splatRGB.atomicAddition(splat);
	}

	/**
	* Transfer the information to an image file.
	* @param filname - Path of the file
	* @param format - Format of the output image. 0 for PPM and 1 for PFM.
	* @param splatScale - Strength of the splat samples
	*/
	__host__ void Camera::sendRenderToImageFile(std::string filename, int format, float splatScale) {

		float* rgbs = new float[3 * filmResolution.area()];

		getRender(rgbs, splatScale);

		Vector2i d = filmResolution.diagonal();

		//outputFile(filename.c_str(), rgbs, d.x, d.y, 0);

		if (format == 0) writeToPPM(filename.c_str(), rgbs, d.x, d.y);
		else writeToPFM(filename.c_str(), rgbs, d.x, d.y);
	}

	__host__ void Camera::getRender(float* rgbs, float splatScale) const {

		Pixel* h_film;

		h_film = (Pixel*)malloc(sizeof(Pixel) * filmResolution.area());
		gpuErrCheck(cudaMemcpy(h_film, d_film, sizeof(Pixel) * filmResolution.area(), cudaMemcpyDeviceToHost));

		Vector2i dimensions = filmResolution.diagonal();

		for (int j = 0; j < dimensions.y; j++) {
			for (int i = 0; i < dimensions.x; i++) {

				int indexFilm = (dimensions.y - 1 - j) * dimensions.x + i;
				int indexRGB = j * dimensions.x + i;

				Pixel currentPixel = h_film[indexFilm];

				float invWeight = 1 / h_film[indexFilm].filterWeight;

				// Add sample rgb
				rgbs[3 * indexRGB] = fmaxf(0.0f, currentPixel.rgb.r * invWeight);
				rgbs[3 * indexRGB + 1] = fmaxf(0.0f, currentPixel.rgb.g * invWeight);
				rgbs[3 * indexRGB + 2] = fmaxf(0.0f, currentPixel.rgb.b * invWeight);

				// Add splat rgb
				rgbs[3 * indexRGB] += currentPixel.splatRGB.r * splatScale;
				rgbs[3 * indexRGB + 1] += currentPixel.splatRGB.g * splatScale;
				rgbs[3 * indexRGB + 2] += currentPixel.splatRGB.b * splatScale;
			}
		}

		free(h_film);
		cudaFree(d_film);
	}
}

#endif