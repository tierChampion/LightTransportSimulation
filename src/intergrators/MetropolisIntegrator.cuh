#ifndef __metropolis_cuh__
#define __metropolis_cuh__

#include "BidirectionalPathIntegrator.cuh"

namespace lts {

	// Need the framework of the bidirectional path tracing integrator.

	class MetropolisIntegrator {

		Camera* h_camera, d_camera;
		// special sampler
		Scene* scene;

	};
}

#endif