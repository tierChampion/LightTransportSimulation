#ifndef __integrator_kernel_cu__
#define __integrator_kernel_cu__

#include "IntegratorHelper.cuh"
#include "PathTracingIntegrator.cuh"
#include "BidirectionalPathIntegrator.cuh"

namespace lts {

	__global__ void lightDistributionKernel(Distribution1D* distribution, const Scene* scene) {

		if (blockIdx.x * blockDim.x + threadIdx.x > 0) return;

		*distribution = computeLightPowerDistribution(*scene);
	}

	__global__ void PathTracingKernel(PathTracingIntegrator* PT_integrator, Distribution1D* l_distrib, unsigned int seed) {

		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;

		PT_integrator->evaluate(i, j, *l_distrib, seed);
	}

	__global__ void BidirectionalPathTracingKernel(BidirectionalPathIntegrator* BDPT_integrator,
		Distribution1D* l_distrib, unsigned int seed) {

		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;

		BDPT_integrator->evaluate(i, j, seed, *l_distrib);
	}
}

#endif