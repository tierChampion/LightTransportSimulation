#ifndef __scene_loader_kernels_cu__
#define __scene_loader_kernels_cu__

#include "SceneLoader.cuh"

namespace lts {

	__global__ void materialInitKernel(Material** materials, char* textureParameters, Mipmap* mpmps,
		int materialCount) {

		// No parallelisation
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index > 0) return;

		int mpmpCounter = 0;

		for (int m = 0; m < materialCount; m++) {

			Texture<Spectrum>* Kd;

			if (textureParameters[m] == 'i') {
				Kd = new ImageTexture(new UVMapping2D(1, 1, 0, 0), &mpmps[mpmpCounter++]);
			}
			else if (textureParameters[m] == 'c') {
				Kd = new ConstantTexture(Spectrum(1.0f));
			}

			materials[m] = new MatteMaterial(Kd, 0.0f);
		}
	}

	__global__ void new_sceneInitKernel(Triangle* tris, Primitive* prims,
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
		prims[triIndex].setMaterial(materials[meshIndex]);

		info->addPrimitive(prims, triIndex);
	}
}

#endif