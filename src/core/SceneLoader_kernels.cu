#ifndef __scene_loader_kernels_cu__
#define __scene_loader_kernels_cu__

#include "SceneLoader.cuh"

namespace lts {

	/*
	* todo:
	* - Put every material seperatly in the file (to simplify the parameter instancing)
	* - Allow many materials
	* - Make kernel parallel for each material
	*/

	__global__ void materialInitKernel(Material** materials,
		char* materialTypes, char* textureTypes, int* matStarts,
		float* materialParams, float* roughnesses,
		Mipmap* mpmps, int materialCount) {

		// No parallelisation
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index > 0) return;

		for (int m = 0; m < materialCount; m++) {

			Texture<Spectrum>* Kd;

			int start = matStarts[m];

			switch (textureTypes[m]) {
			case 'c': Kd = new ConstantTexture(Spectrum(materialParams[start],
				materialParams[start + 1],
				materialParams[start + 2])); break;
			case 'b': Kd = new BilerpTexture(new UVMapping2D(1, 1, 0, 0),
				Spectrum(1.00f, 0.01f, 0.01f),
				Spectrum(0.01f, 1.0f, 0.01f),
				Spectrum(0.01f, 0.0f, 1.01f),
				Spectrum(0.01f, 0.01f, 0.0f)); break; // undone for now, not really useful and really long
			case 'i': Kd = new ImageTexture(new UVMapping2D(1, 1, 0, 0), &mpmps[(int)materialParams[start]]); break; // 0 -> 1?
			case 'f': Kd = new FBmTexture<Spectrum>(Transform(),
				materialParams[start],
				materialParams[start + 1]); break;
			case 'w': Kd = new WrinkledTexture<Spectrum>(Transform(),
				materialParams[start],
				materialParams[start + 1]); break;
			case 'v': Kd = new WindyTexture<Spectrum>(Transform()); break;
			case 'm': Kd = new MarbleTexture(Transform(),
				materialParams[start],
				materialParams[start + 1],
				materialParams[start + 2],
				materialParams[start + 3]); break;
			}

			switch (materialTypes[m]) {
			case 'L': materials[m] = new MatteMaterial(Kd, roughnesses[m]); break;
				//materials[m] = new MetalMaterial(Spectrum(0.01f, 0.01f, 3.068099f), 
				//Spectrum(0.01f, 0.01f, 0.18104f), roughnesses[m]);
			}
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