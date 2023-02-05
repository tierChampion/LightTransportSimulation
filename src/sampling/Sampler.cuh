#ifndef __sampler_cuh__
#define __sampler_cuh__

#include "Sampling.cuh"

namespace lts {

	/**
	* Sampler for generating random variables with specific distributions
	*/
	class Sampler {

		curandState_t* states;
		int strataDimension;

	public:

		int samplesPerPixel;
		float sqrtSPP;

		__host__ Sampler() {}

		/**
		* Create a sampler
		* @param spp - Samples per pixels
		* @param threadCount - number of threads
		*/
		__host__ Sampler(int spp, int threadCount) : samplesPerPixel(spp) {
			gpuErrCheck(cudaMalloc(&states, threadCount * sizeof(curandState_t)));
			sqrtSPP = sqrtf(samplesPerPixel);
			strataDimension = (int)floorf(sqrtSPP);
		}

		/**
		* Initializes the RNG associated with the thread.
		* @param id - thread number
		* @param seed - random seed
		*/
		__device__ void prepareThread(int id, unsigned int seed) {

			curand_init(seed + wangHash(id), 0, 0, &states[id]);
		}

		/**
		* Returns a stratified random variable.
		* @param threadId - also the id of the pixel
		* @param sample - n-th sample of the threadId pixel.
		*/
		__device__ Point2f stratifiedSample(int threadId, int sample) const {

			Point2f omega;

			omega.x = curand_uniform(&states[threadId]);
			omega.y = curand_uniform(&states[threadId]);

			if (sample - 1 < strataDimension * strataDimension) {

				omega /= strataDimension;
				omega += Point2f((sample % strataDimension) / (float)strataDimension,
					(sample / strataDimension) / (float)strataDimension);
			}

			return omega;
		}

		/**
		* Returns a uniformly distributed random 2D vector.
		* @param id - id of the thread
		*/
		__device__ Point2f uniformSample(int id) const {

			return Point2f(curand_uniform(&states[id]), curand_uniform(&states[id]));
		}

		/**
		* Returns a uniformly distributed random variable.
		* @param id - id of the thread
		*/
		__device__ float get1DSample(int id) const {
			return curand_uniform(&states[id]);
		}
	};
}

#endif