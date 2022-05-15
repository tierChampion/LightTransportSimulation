#ifndef __scene_loader_cuh__
#define __scene_loader_cuh__

#include "FileIO.cuh"
#include "../acceleration/BVH.cuh"
#include "../acceleration/Scene.cuh"
#include "../materials/textures/Procedural.cuh"
#include "../materials/textures/ImageTexture.cuh"

namespace lts {

	__global__ void materialInitKernel(Material** materials, char* textureParameters, Mipmap* mpmps,
		int materialCount);

	__global__ void new_sceneInitKernel(Triangle* tris, Primitive* prims,
		BVHPrimitiveInfo* info, MortonPrimitives* morton, TriangleMesh* meshes,
		Material** materials, Light** lights, float* LEs,
		int meshCount, int materialCount, int areaLightCount);

	__host__ inline Scene* parseScene(std::string sceneName) {

		// SEE KERNELBOX.SCENE FOR TEMPLATE EXAMPLE
		// Only support arealights for now
		// Only support black imagewrap for now
		// Only support matte material for now with no roughness
		// Only support image or white constant textures
		// Only support a scene with no subjects

		std::ifstream stream;
		stream.open(sceneName + ".scene");
		std::string token;

		int imgCount;
		std::vector<Mipmap> mpmps;
		int meshCount;
		std::vector<std::string> meshFiles;
		int materialCount;
		std::vector<char> textures;
		int areaLightMeshCount;
		std::vector<float> les;

		if (stream.is_open()) {


			// Number of images or mipmaps
			stream >> token;  imgCount = std::stoi(token);
			// Mipmaps creation
			for (int i = 0; i < imgCount; i++) {
				stream >> token; mpmps.emplace_back(CreateMipMap(token.c_str(), ImageWrap::Black));
			}
			// Number of meshes
			stream >> token; meshCount = std::stoi(token);
			// Mesh files
			for (int m = 0; m < meshCount; m++) {
				stream >> token; meshFiles.emplace_back(token);
			}
			// Number of materials
			stream >> token; materialCount = std::stoi(token);
			// Material parameters
			for (int mat = 0; mat < materialCount; mat++) {
				stream >> token; textures.emplace_back(token[0]);
			}
			// Number of meshes with light
			stream >> token; areaLightMeshCount = std::stoi(token);
			// AreaLight parameters
			for (int al = 0; al < areaLightMeshCount; al++) {
				stream >> token; les.emplace_back(std::stof(token));
				stream >> token; les.emplace_back(std::stof(token));
				stream >> token; les.emplace_back(std::stof(token));
			}
		}
		stream.close();

		/* Meshes initialisation */

		std::vector<TriangleMesh> h_meshes;
		int triCount = 0;
		int highestTriCount = 0;
		int lightCount = 0;

		for (int f = 0; f < meshCount; f++) {

			std::string file = meshFiles[f];

			h_meshes.push_back(*parseMesh(file, Transform()));

			triCount += h_meshes.back().nTriangles;
			if (f >= meshCount - areaLightMeshCount) lightCount += h_meshes.back().nTriangles;
			if (h_meshes.back().nTriangles > highestTriCount) highestTriCount = h_meshes.back().nTriangles;
		}
		TriangleMesh* d_meshes = passToDevice(h_meshes.data(), meshCount);

		/* Materials initialisation */
		Material** d_materials;
		Mipmap* d_mpmps = passToDevice(mpmps.data(), imgCount);
		char* d_params = passToDevice(textures.data(), materialCount);
		gpuErrCheck(cudaMalloc(&d_materials, sizeof(Material**)));

		// Material kernel
		materialInitKernel << <1, 1 >> > (d_materials, d_params, d_mpmps, materialCount);
		gpuErrCheck(cudaDeviceSynchronize());
		gpuErrCheck(cudaPeekAtLastError());

		/* Primitives & Acceleration Structure allocation and initialisation */

		// Acceleration structure
		BVHPrimitiveInfo* h_info = new BVHPrimitiveInfo(triCount);
		BVHPrimitiveInfo* d_info = passToDevice(h_info);
		MortonPrimitives* h_morton = new MortonPrimitives(triCount);
		MortonPrimitives* d_morton = passToDevice(h_morton);
		// Primitives
		Triangle* d_tris;
		Primitive* d_prims;
		Light** d_lights;
		float* d_les = passToDevice(les.data(), 3 * areaLightMeshCount);
		gpuErrCheck(cudaMalloc(&d_tris, (triCount) * sizeof(Triangle)));
		gpuErrCheck(cudaMalloc(&d_prims, (triCount) * sizeof(Primitive)));
		gpuErrCheck(cudaMalloc(&d_lights, sizeof(Light**)));

		dim3 block(BLOCK_SIZE, 1);
		dim3 grid(highestTriCount / BLOCK_SIZE +
			(highestTriCount % BLOCK_SIZE != 0), meshCount);

		// scene init kernel
		new_sceneInitKernel << <grid, block >> > (d_tris, d_prims, d_info, d_morton,
			d_meshes, d_materials, d_lights, d_les, meshCount, materialCount, areaLightMeshCount);
		gpuErrCheck(cudaDeviceSynchronize());
		gpuErrCheck(cudaPeekAtLastError());
		// morton kernel
		calculateMortonCodes << <grid, block >> > (d_info, d_morton);
		gpuErrCheck(cudaDeviceSynchronize());
		gpuErrCheck(cudaPeekAtLastError());

		BVH* d_tree = CreateBVHTree(d_prims, d_info, h_morton, d_morton, triCount);
		cudaFree(d_info);
		delete h_info;
		cudaFree(d_morton);
		delete h_morton;

		Scene* scene = new Scene(d_tree, d_lights, meshCount, lightCount);

		return passToDevice(scene);
	}
}

#endif