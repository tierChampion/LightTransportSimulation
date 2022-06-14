#ifndef __scene_loader_kernels_cu__
#define __scene_loader_kernels_cu__

#include "SceneLoader.cuh"

namespace lts {

	__global__ void materialInitKernel(Material** materials,
		char* materialTypes, char* textureTypes, int* matStarts, int* texStarts,
		float* materialParams, float* extras,
		Mipmap* mpmps, int materialCount) {

		int m = blockIdx.x * blockDim.x + threadIdx.x;
		if (m > materialCount) return;

		if (materialTypes[m] != 'M') {

			int matStart = matStarts[m];
			int start = texStarts[matStart];

			if (materialTypes[m] == 'L') {
				Texture<Spectrum>* Kd;
				setMatTexture(&Kd, mpmps, textureTypes, materialParams,
					matStart, start);
				materials[m] = new MatteMaterial(Kd, extras[m]);
			}
			else if (materialTypes[m] == 'R') {
				// Reflected
				Texture<Spectrum>* R;
				setMatTexture(&R, mpmps, textureTypes, materialParams,
					matStart, start);
				materials[m] = new MirrorMaterial(R);
			}
			else {
				// Reflected
				Texture<Spectrum>* R;
				setMatTexture(&R, mpmps, textureTypes, materialParams,
					matStart, start);
				// Transmitted
				matStart++; start = texStarts[matStart];
				Texture<Spectrum>* T;
				setMatTexture(&T, mpmps, textureTypes, materialParams,
					matStart, start);

				materials[m] = new GlassMaterial(extras[m], R, T);
			}
			//materials[m] = new MetalMaterial(Spectrum(0.01f, 0.01f, 3.068099f), 
			//Spectrum(0.01f, 0.01f, 0.18104f), roughnesses[m]);
		}
	}

	__global__ void sceneInitKernel(Triangle* tris, Primitive* prims,
		BVHPrimitiveInfo* info, MortonPrimitives* morton, TriangleMesh* meshes,
		Material** materials, Light** lights, float* LEs,
		int meshCount, int materialCount, int areaLightCount) {

		// Index-th triangle in the current mesh
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		// MeshIndex-th mesh
		int meshIndex = blockIdx.y;

		if (meshIndex >= meshCount) return;
		if (index >= meshes[meshIndex].nTriangles) return;

		TriangleMesh* mesh = &meshes[meshIndex];
		// TriIndex-th triangle in the total scene
		int triIndex = index;
		for (int m = 0; m < meshIndex; m++) {
			triIndex += meshes[m].nTriangles;
		}

		tris[triIndex] = Triangle(mesh, index);

		AreaLight* al = nullptr;

		if (meshIndex >= meshCount - areaLightCount) {
			int leIndex = meshCount - meshIndex - 1;
			Spectrum Le = Spectrum(LEs[leIndex * 3], LEs[leIndex * 3 + 1], LEs[leIndex * 3 + 2]);
			lights[index] = new AreaLight(Transform(), Le, &tris[triIndex]);
			al = (AreaLight*)lights[index];
		}

		prims[triIndex] = Primitive(&tris[triIndex], al);
		prims[triIndex].setMaterial(materials[meshIndex]); // materials cant be reused

		info->addPrimitive(prims, triIndex);
	}

	__global__ void infiniteLightsKernel(BVHPrimitiveInfo* info, Light** lights,
		float* LEs, TriangleMesh* meshes, int areaLightCount, int meshCount) {

		int index = blockIdx.x * blockDim.x + threadIdx.x;

		if (index > 0) return;

		Spectrum L = Spectrum(LEs[3 * areaLightCount],
			LEs[3 * areaLightCount + 1],
			LEs[3 * areaLightCount + 2]);

		InfiniteLight* infiniteL = new InfiniteLight(Transform(), L);
		infiniteL->preprocess(info->bounds[0]);

		int infiniteIndex = 0;

		for (int l = 1; l <= areaLightCount; l++) {
			infiniteIndex += meshes[meshCount - l].nTriangles;
		}

		lights[infiniteIndex] = infiniteL;
	}
}

#endif