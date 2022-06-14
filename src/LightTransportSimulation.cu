#include <iostream>

#include "rendering/Camera.cuh"
#include "rendering/Spectrum.cuh"
#include "sampling/Sampler.cuh"
#include "core/SceneLoader.cuh"
#include "lights/VisibilityTester.cuh"
#include "intergrators/PathTracingIntegrator.cuh"
#include "intergrators/BidirectionalPathIntegrator.cuh"

using namespace lts;

/*
* todo:
*
*	add comments to a bunch of files to make them more readable
*/

const static int RENDER_PIXEL_WIDTH = 1920;
const static int RENDER_PIXEL_HEIGHT = 1080;
const static int SAMPLE_PER_PIXEL = 1;
const static IntegratorType RENDERING_STRATEGY = IntegratorType::PathTracing;
const static int MAX_BOUNCE = 8;
const static int ROULETTE_START = 2;
const static bool PPM_FORMAT = true;

std::string outputFileWithoutExtension("outputs\\test");
const static std::string OUTPUT_FILE = outputFileWithoutExtension + (PPM_FORMAT ? ".ppm" : "pfm");

const char* SUBJECT_FILE = "res/holder";
const char* SCENE_FILE = "res/platforms/bigBox";

int main() {

	int version;
	cudaRuntimeGetVersion(&version);
	printf("---Used CUDA version: %i---\n", version);

	cudaDeviceSetLimit(cudaLimitStackSize, 8192);

	std::cout << "===RENDERING PARAMETERS===\n" <<
		"	Technique: " << toString(RENDERING_STRATEGY) << "\n" <<
		"	Image resolution: " << RENDER_PIXEL_WIDTH << "x" << RENDER_PIXEL_HEIGHT << "\n" <<
		"	Samples per pixel: " << SAMPLE_PER_PIXEL << "\n" <<
		"	Maximum bounces: " << MAX_BOUNCE << std::endl;

	// Camera initialisation
	Filter* f = new GaussianFilter(Vector2f(1.0f, 1.0f), 1.0f);
	Camera* h_cam;

	// Scene initialisation
	auto start = std::chrono::high_resolution_clock::now();
	Scene* scene = parseScene(&h_cam, f,
		0.f, RENDER_PIXEL_WIDTH, RENDER_PIXEL_HEIGHT,
		SCENE_FILE, SUBJECT_FILE);
	delete f;
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << "(1) Scene creation finished in " <<
		duration.count() / 1000.0f << " seconds." << std::endl;

	// Light distribution initialisation
	Distribution1D* d_distribution;
	gpuErrCheck(cudaMalloc((void**)&d_distribution, sizeof(Distribution1D)));
	lightDistributionKernel << <1, 1 >> > (d_distribution, scene);
	gpuErrCheck(cudaDeviceSynchronize());
	gpuErrCheck(cudaPeekAtLastError());

	// Thread count parameters
	dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid = dim3(RENDER_PIXEL_WIDTH / BLOCK_SIZE + (RENDER_PIXEL_WIDTH % BLOCK_SIZE != 0),
		RENDER_PIXEL_HEIGHT / BLOCK_SIZE + (RENDER_PIXEL_HEIGHT % BLOCK_SIZE != 0));

	std::cout << "	Rendering started with grid of " << grid.x << "x" << grid.y << " with block of "
		<< block.x << "x" << block.y << std::endl;

	if (RENDERING_STRATEGY == IntegratorType::PathTracing) {
		// Sampler initialisation
		Sampler h_samp = Sampler(SAMPLE_PER_PIXEL, RENDER_PIXEL_WIDTH * RENDER_PIXEL_HEIGHT);
		Sampler* d_samp = passToDevice(&h_samp);

		// Initialise integrator
		PathTracingIntegrator h_integrator(MAX_BOUNCE, ROULETTE_START, h_cam, d_samp, scene);
		PathTracingIntegrator* d_integrator = passToDevice(&h_integrator);

		// Rendering
		start = std::chrono::high_resolution_clock::now();
		PathTracingKernel << <grid, block >> > (d_integrator, d_distribution, time(0));
		gpuErrCheck(cudaDeviceSynchronize());
		gpuErrCheck(cudaPeekAtLastError());
		end = std::chrono::high_resolution_clock::now();
		duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		std::cout << "(2) Ray tracing finished in " <<
			duration.count() / 1000.0f << " seconds." << std::endl;

		// Saving to image file
		h_integrator.outputResultAndOpen(OUTPUT_FILE, !PPM_FORMAT);
	}

	/**
	* Bidirectional path tracing is not functional for now:
	* Memory usage is a lot more than standard path tracing and the available memory
	* cant handle the needs
	*/
	else if (RENDERING_STRATEGY == IntegratorType::BiderectionalPathTracing) {

		// Sampler initialisation
		Sampler h_samp = Sampler(SAMPLE_PER_PIXEL, RENDER_PIXEL_WIDTH * RENDER_PIXEL_HEIGHT);
		Sampler* d_samp = passToDevice(&h_samp);

		// Initialise integrator
		BidirectionalPathIntegrator h_integrator(MAX_BOUNCE, ROULETTE_START, h_cam, d_samp, scene);
		BidirectionalPathIntegrator* d_integrator = passToDevice(&h_integrator);

		// Rendering
		start = std::chrono::high_resolution_clock::now();
		BidirectionalPathTracingKernel << <grid, block >> > (d_integrator, d_distribution, time(0));
		gpuErrCheck(cudaDeviceSynchronize());
		gpuErrCheck(cudaPeekAtLastError());
		end = std::chrono::high_resolution_clock::now();
		duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		std::cout << "(2) Ray tracing finished in " <<
			duration.count() / 1000.0f << " seconds." << std::endl;

		// Saving to image file
		h_integrator.outputResultAndOpen(OUTPUT_FILE, !PPM_FORMAT);
	}

	return 0;
}