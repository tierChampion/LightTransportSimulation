#ifndef __scene_cuh__
#define __scene_cuh__

#include "../acceleration/BVH.cuh"
#include "../materials/Material.cuh"
#include "../core/FileIO.cuh"

namespace lts {

	class Scene {

		BVH* traversalTree;
		int meshCount;

	public:

		Light** lights;
		int lightCount;

		__host__ Scene(BVH* d_bvh, Light** d_lights, int mCount, int lCount) : traversalTree(d_bvh),
			lights(d_lights),
			meshCount(mCount),
			lightCount(lCount)
		{}

		__device__ bool simpleIntersect(const Ray& r) const {
			return traversalTree->simpleIntersect(r);
		}

		__device__ bool intersect(const Ray& r, SurfaceInteraction* si) const {
			return traversalTree->intersect(r, si);
		}

	};

	/*
	__global__ void sceneInitKernel(Triangle* tris, Primitive* prims,
		BVHPrimitiveInfo* info, MortonPrimitives* morton, TriangleMesh* meshes,
		Material** materials, Light** lights, int meshCount);
		*/
}

#endif