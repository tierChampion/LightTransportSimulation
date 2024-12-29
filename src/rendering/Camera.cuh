#ifndef __camera_cuh__
#define __camera_cuh__

#include "../geometry/Ray.cuh"
#include "Spectrum.cuh"
#include "../lights/VisibilityTester.cuh"
#include "Filter.cuh"
#include "../sampling/Sampling.cuh"

namespace lts {

#define FILTER_TABLE_SIZE 16

	/**
	* Single pixel in the display.
	*/
	struct Pixel {
		Spectrum rgb;
		float filterWeight;
		Spectrum splatRGB;
		float pad; // makes a cleaner size for the pixel
	};

	/**
	* Camera from which the scene is rendered.
	*/
	class Camera {

	public:

		// Geometric representation
		Point3f o;
		Vector3f right;
		Vector3f up;
		Point3f llc;
		Vector3f u, v, w;
		float lensRadius;
		float focalDist;

		// Film representation
		Pixel* d_film;
		Bounds2i filmResolution;
		// Filter representation
		float* d_table;
		Vector2f filterRadius;
		Vector2f invFilterRadius;

		__host__ __device__ Camera() {}

		/**
		* Create a camera with the desired properties.
		* @param lookFrom - Location of the camera
		* @param lookAt - Point to focus on
		* @param vUp - Up direction from the camera
		* @param vertFov - Vertical field of view
		* @param aperture - Size of the lens
		* @param focusDist - Distance at which the camera focuses
		* @param imgWidth - Width of the output image
		* @param imgHeight - Height of the output image
		* @param filter - Filter function used to antialias the output
		*/
		__host__ Camera(const Point3f& lookFrom, const Point3f& lookAt, const Vector3f& vUp,
			float vertFov, float aperture, float focusDist, int imgWidth, int imgHeight, Filter* filter);

		/**
		* Generate the desired ray from the samples.
		* @param pFilm - Sample on the film
		* @param pLens - Sample on the lens
		* @return Ray passing by both samples
		*/
		__device__ Ray generateRay(const Point2f& pFilm, const Point2f& pLens) const;

		/**
		* Generate a ray with differential information on the film.
		* @param pFilm - Sample on the film
		* @param pLens - Sample on the lens
		*/
		__device__ Ray generateDifferentialRay(const Point2f& pFilm, const Point2f& pLens) const;

		/**
		* Evaluates the importance of a given ray
		* @param r - ray to evaluate the importance of
		* @param pRaster2 - location on the raster of the ray. optional
		*/
		__device__ Spectrum we(const Ray& r, Point2f* pRaster2) const;

		__device__ void PdfWe(const Ray& r, float* pdfPos, float* pdfDir) const;

		__device__ Spectrum sampleWi(const Interaction& it, const Point2f sample,
			Vector3f* wi, float* pdf, Point2f* pRaster, VisibilityTester* vis) const;

		/**
		* Adds a sample on the film with filtering.
		* @param pFilm - location on the film. (range [0, filterWidth])
		* @param L - radiance of the sample
		*/
		__device__ void addSample(const Point2f& pFilm, const Spectrum L);

		/**
		* Adds a splat sample on the film without filtering.
		* @param pFilm - location on the film. (range [0, filterWidth])
		* @param L - radiance of the sample
		*/
		__device__ void addSplat(const Point2f& pFilm, const Spectrum splat);

		/**
		* Transfer the information to an image file.
		* @param filname - Path of the file
		* @param format - Format of the output image. 0 for PPM and 1 for PFM.
		* @param splatScale - Strength of the splat samples
		*/
		__host__ void sendRenderToImageFile(std::string filename, int format, float splatScale = 0.0f);

	private:

		__host__ void getRender(float* rgbs, float splatScale) const;
	};
}

#endif