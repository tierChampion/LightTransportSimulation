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
*  add infinite light types (on that samples along the normal, random or always up)
*
*	add comments to a bunch of files to make them more readable
*/

const static IntegratorType RENDERING_STRATEGY = IntegratorType::PathTracing;

int main(int argc, char** argv) {

	std::cout << "$$\\       $$\\            $$\\        $$\\      $$$$$$\\  $$\\" << std::endl;
	std::cout << "$$ |      \\__|           $$ |       $$ |    $$  __$$\\ \\__|" << std::endl;
	std::cout << "$$ |      $$\\   $$$$$$\\  $$$$$$$\\ $$$$$$\\   $$ / \\__| $$\\ $$$$$$\\$$$$\\" << std::endl;
	std::cout << "$$ |      $$ | $$  __$$\\ $$  __$$\\\\_$$  _|  \\$$$$$$\\  $$ |$$  _$$  _$$\\" << std::endl;
	std::cout << "$$ |      $$ | $$ /  $$ |$$ |  $$ | $$ |     \\____$$\\ $$ |$$ / $$ / $$ |" << std::endl;
	std::cout << "$$ |      $$ | $$ |  $$ |$$ |  $$ | $$ |$$\\ $$\\   $$ |$$ |$$ | $$ | $$ |" << std::endl;
	std::cout << "$$$$$$$$\\ $$ | \\$$$$$$$ |$$ |  $$ | \\$$$$  |\\$$$$$$  |$$ |$$ | $$ | $$ |" << std::endl;
	std::cout << "\\________|\\__|  \\____$$ |\\__|  \\__|  \\____/  \\______/ \\__|\\__| \\__| \\__|" << std::endl;
	std::cout << "               $$\\   $$ |" << std::endl;
	std::cout << "               \\$$$$$$  |" << std::endl;
	std::cout << "                \\______/" << std::endl << std::endl;

	int renderWidth, renderHeight;
	int samplesPerPixel;
	int maxBounce;
	int rouletteStart;
	int format;
	std::string outputFile;
	std::string sceneFile;
	std::string subjectFile;

	std::string path = getProjectDir(argv[0], "LightTransportSimulation");

	readApplicationParameters(path + "app.params", &renderWidth, &renderHeight,
		&samplesPerPixel, &maxBounce, &rouletteStart, &format, &outputFile, &sceneFile, &subjectFile);

	int version;
	cudaRuntimeGetVersion(&version);
	printf("---Used CUDA version: %i---\n", version);

	cudaDeviceSetLimit(cudaLimitStackSize, 8192);

	std::cout << "===RENDERING PARAMETERS===\n" <<
		"	Technique: " << toString(RENDERING_STRATEGY) << "\n" <<
		"	Image resolution: " << renderWidth << "x" << renderHeight << "\n" <<
		"	Samples per pixel: " << samplesPerPixel << "\n" <<
		"	Maximum bounces: " << maxBounce << std::endl;

	// Camera initialisation
	Filter* f = new GaussianFilter(Vector2f(1.0f, 1.0f), 1.0f);
	Camera* h_cam;

	// Scene initialisation
	auto start = std::chrono::high_resolution_clock::now();
	Scene* scene = parseScene(&h_cam, f,
		0.f, renderWidth, renderHeight,
		path, sceneFile, subjectFile);
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
	dim3 grid = dim3(renderWidth / BLOCK_SIZE + (renderWidth % BLOCK_SIZE != 0),
		renderHeight / BLOCK_SIZE + (renderHeight % BLOCK_SIZE != 0));

	std::cout << "	Rendering started with grid of " << grid.x << "x" << grid.y << " with block of "
		<< block.x << "x" << block.y << std::endl;

	if (RENDERING_STRATEGY == IntegratorType::PathTracing) {
		// Sampler initialisation
		Sampler h_samp = Sampler(samplesPerPixel, renderWidth * renderHeight);
		Sampler* d_samp = passToDevice(&h_samp);

		// Initialise integrator
		PathTracingIntegrator h_integrator(maxBounce, rouletteStart, h_cam, d_samp, scene);
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
		h_integrator.outputResultAndOpen(path + outputFile, format);
	}

	/**
	* Bidirectional path tracing is not functional for now:
	* Memory usage is a lot more than standard path tracing and the available memory
	* cant handle the needs
	*/
	else if (RENDERING_STRATEGY == IntegratorType::BiderectionalPathTracing) {

		// Sampler initialisation
		Sampler h_samp = Sampler(samplesPerPixel, renderWidth * renderHeight);
		Sampler* d_samp = passToDevice(&h_samp);

		// Initialise integrator
		BidirectionalPathIntegrator h_integrator(maxBounce, rouletteStart, h_cam, d_samp, scene);
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
		h_integrator.outputResultAndOpen(outputFile, format);
	}

	return 0;
}