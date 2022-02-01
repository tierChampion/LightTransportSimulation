#ifndef __bvh_kernels_cuh__
#define __bvh_kernels_cuh__

#include "BVH.cuh"

namespace lts {

	__global__ void calculateMortonCodes(BVHPrimitiveInfo* info, MortonPrimitives* mortons) {

		int index = blockIdx.x * blockDim.x + threadIdx.x;

		if (index >= info->count) return;

		mortons->convertInfoToMorton(info, index);
	}

	__global__ void createBuildWithoutBounds(MortonPrimitives* keys,
		LBVHBuildNode* leaf, LBVHBuildNode* interior) {

		// fetch index
		int index = blockIdx.x * blockDim.x + threadIdx.x;

		if (index >= keys->count) return;

		// create node with treeBuilding
		treeBuilding(keys, leaf, interior, index);
	}

	__global__ void addBoundsToBuild(BVHPrimitiveInfo* info,
		LBVHBuildNode* leaf) {

		// fetch index
		int index = blockIdx.x * blockDim.x + threadIdx.x;

		if (index >= info->count) return;

		// add bounds with addBoundsToNode
		addBoundsToNode(info, leaf, index);
	}

	__global__ void createBVH(LBVHBuildNode* leaf, LBVHBuildNode* interior, BVH* tree) {

		// fetch index
		int index = blockIdx.x * blockDim.x + threadIdx.x;

		if (index - 1 >= (tree->count - 1) / 2) return;

		// transform build node to traversal node with setTraversalTree
		tree->setTraversalTree(leaf, interior, index);
	}
}

#endif