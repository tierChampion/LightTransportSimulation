#include <iostream>

#include "geometry/Transform.cuh"
#include "shapes/Triangle.cuh"
#include "rendering/Camera.cuh"
#include "rendering/Spectrum.cuh"
#include "sampling/Sampler.cuh"
#include "acceleration/BVH.cuh"
#include "acceleration/Scene.cuh"
#include "core/FileIO.cuh"
#include "lights/VisibilityTester.cuh"
#include "materials/textures/Procedural.cuh"
#include "materials/textures/ImageTexture.cuh"
#include "intergrators/PathTracingIntegrator.cuh"
#include "intergrators/BidirectionalPathIntegrator.cuh"
#include "core/ImageI.cuh"

using namespace lts;

/*
* Things to do next:
*
* 1. Implement other rendering algorithms
*
* 2. Implement image textures.
*
* x3. Create a blocked array for the mip mapping of image textures.
*/

const static int RENDER_PIXEL_WIDTH = 512;
const static int RENDER_PIXEL_HEIGHT = 512;
const static int SAMPLE_PER_PIXEL = 1;
const static IntegratorType RENDERING_STRATEGY = IntegratorType::PathTracing;
const static int MAX_BOUNCE = 10;
const static int ROULETTE_START = 2;
const static bool PPM_FORMAT = true;

std::string outputFileWithoutExtension("outputs\\test");
const static std::string OUTPUT_FILE = outputFileWithoutExtension + (PPM_FORMAT ? ".ppm" : "pfm");

const char* SUBJECT_FILE = "res/textureHolder";

__host__ inline
BVH* BVHTreeCreationKernel(Primitive* prims, BVHPrimitiveInfo* info,
	MortonPrimitives* h_mortons, MortonPrimitives* d_mortons,
	int primCount) {

	sortMortonPrims(h_mortons);

	dim3 block(BLOCK_SIZE, 1);
	dim3 grid(primCount / BLOCK_SIZE + (primCount % BLOCK_SIZE != 0), 1);

	// try and separate leaf and interior
	LBVHBuildNode* h_leaf = new LBVHBuildNode[primCount];
	LBVHBuildNode* h_interior = new LBVHBuildNode[primCount - 1];
	for (int k = 0; k < primCount - 1; k++) h_interior[k].key = k;
	for (int l = 0; l < primCount; l++) h_leaf[l].key = l;

	LBVHBuildNode* d_interior = passToDevice(h_interior, primCount - 1);
	LBVHBuildNode* d_leaf = passToDevice(h_leaf, primCount);

	createBuildWithoutBounds << <grid, block >> > (d_mortons, d_leaf, d_interior);
	gpuErrCheck(cudaDeviceSynchronize());
	gpuErrCheck(cudaPeekAtLastError());

	addBoundsToBuild << <grid, block >> > (info, d_leaf);
	gpuErrCheck(cudaDeviceSynchronize());
	gpuErrCheck(cudaPeekAtLastError());

	BVH* h_traversalTree = new BVH(prims, 2 * primCount - 1);
	BVH* d_traversalTree = passToDevice(h_traversalTree);

	createBVH << <grid, block >> > (d_leaf, d_interior, d_traversalTree);
	gpuErrCheck(cudaDeviceSynchronize());
	gpuErrCheck(cudaPeekAtLastError());

	cudaFree(d_interior);
	delete[] h_interior;
	cudaFree(d_leaf);
	delete[] h_leaf;
	delete h_traversalTree;

	return d_traversalTree;
}

// todo
__host__ inline Mipmap mipmapInitialisation(const char* file, ImageWrap wrapMode) {

	int width, height;
	Spectrum* img = loadImageFile_s(file, &width, &height);
	Spectrum* d_img = passToDevice(img, width * height);

	int level = log2f(fmaxf(width, height));

	BlockedArray<Spectrum>* d_pyr;
	gpuErrCheck(cudaMalloc((void**)&d_pyr, level * sizeof(BlockedArray<Spectrum>)));

	dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid;
	int w = width, h = height;

	for (int l = 0; l < level; l++) {

		if (l == 0) {
			pyramidBaseInit << <1, 1 >> > (d_pyr, width, height, d_img, level);
		}
		else {

			grid = dim3(w / BLOCK_SIZE + (w % BLOCK_SIZE != 0),
				h / BLOCK_SIZE + (h % BLOCK_SIZE != 0));

			pyramidInitKernel << <grid, block >> > (d_pyr, l, wrapMode);
		}
		gpuErrCheck(cudaDeviceSynchronize());
		gpuErrCheck(cudaPeekAtLastError());
		w = fmaxf(1, w / 2);
		h = fmaxf(1, h / 2);
	}

	return Mipmap(wrapMode, d_pyr, width, height, level);
}

/* Scene creation kernels */

__global__ void materialKernel(Material** materials, Light** lights, Mipmap* mpmp) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index > 0) return;

	materials[0] = new MatteMaterial(new ImageTexture(new UVMapping2D(1, 1, 0, 0), mpmp), 0.0f);
	materials[1] = new MatteMaterial(new ConstantTexture<Spectrum>(Spectrum(1.0f)), 0.0f);
	materials[2] = new MatteMaterial(new ConstantTexture<Spectrum>(Spectrum(0.0f, 0.5f, 0.0f)), 0.0f);
	materials[3] = new MatteMaterial(new ConstantTexture<Spectrum>(Spectrum(0.5f, 0.0f, 0.0f)), 0.0f);
}

__global__ void sceneInitKernel(Triangle* tris, Primitive* prims,
	BVHPrimitiveInfo* info, MortonPrimitives* morton, TriangleMesh* meshes,
	Material** materials, Light** lights, int meshCount) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int meshIndex = blockIdx.y;

	if (meshIndex >= meshCount) return;
	if (index >= meshes[meshIndex].nTriangles) return;

	TriangleMesh* mesh = &meshes[meshIndex];
	int triIndex = index;

	for (int m = 0; m < meshIndex; m++) {
		triIndex += meshes[m].nTriangles;
	}

	tris[triIndex] = Triangle(mesh, index);

	if (meshIndex == meshCount - 1) {
		lights[index] = new AreaLight(Transform(), Spectrum(30.0f), 1, &tris[triIndex]);
	}

	prims[triIndex] = Primitive(&tris[triIndex], meshIndex == meshCount - 1 ? (AreaLight*)lights[index] : nullptr);
	prims[triIndex].setMaterial(materials[meshIndex < meshCount - 1 ? meshIndex : meshIndex - 1]);

	info->addPrimitive(prims, triIndex);
}

__host__ inline
Scene* sceneCreationKernel(std::vector<const char*> meshFiles, Mipmap* d_mpmp) {

	Triangle* tris;
	Primitive* prims;
	std::vector<TriangleMesh> h_meshes;

	int meshCount = 0;
	int triCount = 0;
	int highestTriCount = 0;

	for (const char* file : meshFiles) {

		h_meshes.push_back(*parseMesh(file));

		meshCount++;
		triCount += h_meshes.back().nTriangles;
		if (h_meshes.back().nTriangles > highestTriCount) highestTriCount = h_meshes.back().nTriangles;
	}

	TriangleMesh* d_meshes = passToDevice(h_meshes.data(), meshCount);

	Material** d_materials;
	int matCount = 4;
	gpuErrCheck(cudaMalloc(&d_materials, sizeof(Material**)));
	Light** d_lights;
	int lightCount = 2;
	gpuErrCheck(cudaMalloc(&d_lights, sizeof(Light**)));

	materialKernel << <1, 1 >> > (d_materials, d_lights, d_mpmp);
	gpuErrCheck(cudaDeviceSynchronize());
	gpuErrCheck(cudaPeekAtLastError());

	// Scene & Acceleration allocation and initialisation
	BVHPrimitiveInfo* h_info = new BVHPrimitiveInfo(triCount);
	BVHPrimitiveInfo* d_info = passToDevice(h_info);

	MortonPrimitives* h_morton = new MortonPrimitives(triCount);
	MortonPrimitives* d_morton = passToDevice(h_morton);
	// Primitives allocation
	gpuErrCheck(cudaMalloc(&tris, (triCount) * sizeof(Triangle)));
	gpuErrCheck(cudaMalloc(&prims, (triCount) * sizeof(Primitive)));

	dim3 block(BLOCK_SIZE, 1);
	dim3 grid(highestTriCount / BLOCK_SIZE +
		(highestTriCount % BLOCK_SIZE != 0), meshCount);

	sceneInitKernel << <grid, block >> > (tris, prims, d_info, d_morton, d_meshes, d_materials, d_lights, meshCount);
	gpuErrCheck(cudaDeviceSynchronize());
	gpuErrCheck(cudaPeekAtLastError());

	calculateMortonCodes << <grid, block >> > (d_info, d_morton);
	gpuErrCheck(cudaDeviceSynchronize());
	gpuErrCheck(cudaPeekAtLastError());

	BVH* d_tree = BVHTreeCreationKernel(prims, d_info, h_morton, d_morton, triCount);
	cudaFree(d_info);
	delete h_info;
	cudaFree(d_morton);
	delete h_morton;

	Scene* scene = new Scene(d_tree, d_lights, meshCount, lightCount);

	return passToDevice(scene);
}

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

int main() {

	Mipmap mpmp = mipmapInitialisation("res/yo!.png", ImageWrap::Black); // fix mip map creation
	Mipmap* d_mpmp = passToDevice(&mpmp);

	cudaDeviceSetLimit(cudaLimitStackSize, 8192);

	std::cout << "Rendering parameters: \n" <<
		"Technique: " << toString(RENDERING_STRATEGY) << "\n" <<
		"Image resolution: " << RENDER_PIXEL_WIDTH << "x" << RENDER_PIXEL_HEIGHT << " pixels\n" <<
		"Samples per pixel: " << SAMPLE_PER_PIXEL << "\n" <<
		"Maximum bounces: " << MAX_BOUNCE << std::endl;

	// Camera initialisation
	Filter* f = new LanczosSincFilter(Vector2f(1.0f, 1.0f), 1.0f);
	Camera* h_cam = new Camera(Point3f(0.0f, 0.0f, -28.0f), Point3f(0, 0, 0), Vector3f(0, 1, 0), 40.0f, 0.0f, 28.0f,
		RENDER_PIXEL_WIDTH, RENDER_PIXEL_HEIGHT, f);
	delete f;

	std::vector<const char*> objFiles;
	// object to see in scene
	objFiles.push_back(SUBJECT_FILE);
	// scene
	objFiles.push_back("res/kernel_box_scene/kernelBox_gray");
	objFiles.push_back("res/kernel_box_scene/kernelBox_green");
	objFiles.push_back("res/kernel_box_scene/kernelBox_red");
	// light
	objFiles.push_back("res/kernel_box_scene/kernelBox_light");

	// Scene initialisation
	auto start = std::chrono::high_resolution_clock::now();
	Scene* scene = sceneCreationKernel(objFiles, d_mpmp);
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

	std::cout << "Rendering started with grid of " << grid.x << "x" << grid.y << " with block of "
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