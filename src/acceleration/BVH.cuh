#ifndef __bvh_cuh__
#define __bvh_cuh__

#include "Primitive.cuh"
#include "../geometry/Geometry.cuh"

namespace lts {

	/*
	* Linear bvh construction:
	*
	* The tree will contain two list of nodes:
	* 1. Interior nodes (branch nodes)
	* 2. Leaf nodes
	*
	* The leaf nodes are simply going to be a morton code and an index towards the primitive or
	* somekind of information about the primitive.
	*
	* The interior nodes is going to have a key that represents the splitting point of the interval or the greatest different bit.
	*
	* This means there will be the same number of interior nodes and that they will also have a great property: the key of
	* an interior node will be the same as either the last or the first key in the interval.
	* For a split position x in a range [i, j], the left child will be x with the interval [i, x] and the right child will be
	* x + 1 with the interval [x + 1, j]
	*
	* This also means that any leaf node will have a corresponding unique interior node, which can be found independently of the rest
	* of the tree! This means the rest can be done fully parallel.
	*
	* Position will be encoded in morton code with the following construction:
	* for p = x, y, z
	*
	* Algo :
	*
	* 0. Data need to be processed. All the primitives in the scene must be stored as both a world bound as well as a
	* centroid, which is the center of the bound. The centroid is then converted to morton code to be used in the rest
	* of the algorithm.
	*
	* 1. Determine the direction of the interval. Either the direction is negative, which means the current node is the
	* end of its interval or the direction is positive and the current node is the beginning. The direction d
	* is calculated by doing (delta(i, i + 1) - delta(i, i - 1)) > 0 ? 1 : -1 where delta(x, y) returns the position of the first
	* different bit.
	* EX: <<< for i( 010 ), i - 1( 001) and i + 1( 100 ) -> delta(i, i + 1) - delta(i, i - 1) = 0 - 1 = -1 >>>
	*
	* 2. After finding d, variables can now be defined. Since normally every i - xd will have a smaller different bit than
	* i + yd for any positive x and y, a deltaMin = delta(i, i - d) must be smaller than any delta in the interval of the interior node.
	* EX (WITH SAME VALUES AS ABOVE): <<< d = -1 -> deltaMin = delta(i, i + 1) = 0 -> delta(i, j) > deltaMin for j in node >>>
	*
	* 3. Now, to find the length of the interval, the same principle can be used. First a value lMax is found such that delta(i, i + ld)
	* respects the inequality with deltaMin. lMax starts at 2 and progressively increases by a factor of 2. The real length l can
	* then be found by binary search with the interval [i, i + lMax * d]. After the length found, the other side of the interval can
	* thus be calculated with j = i + l * d.
	*
	* 4. With the range of the node deltaNode = delta(i, j), the same tactic used to find l can be used to find s such that
	* it is the largest value with delta(i, i + s * d) > deltaNode. The split position gamma is then equal to i + s * d + min(d, 0).
	* The range of the children of the node are now known with c1 = [min(i, j), gamma] and c2 = [gamma + 1, max(i, j)].
	* If the range of a child is 1, this means it is a leaf, otherwise, it is an interior node.
	*
	* With these four steps, each interior node can then determine both of his child without knowing his own parent node.
	* The tree is thus constructed and can now be linearized for faster access!
	*
	* Binairy search (for maxRange lMax):
	*
	* l = 0;
	* for (int t = lMax / 2; t > 1; t / 2) {
	*	if (delta(i, i + (l + t) * d) > deltaMin)
	*		l += t;
	* }
	*
	* source 1 with nvidia: https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/
	* source 2 with pbrt: https://www.pbr-book.org/3ed-2018/Primitives_and_Intersection_Acceleration/Bounding_Volume_Hierarchies
	* source 3 for parallel version: https://developer.nvidia.com/blog/parallelforall/wp-content/uploads/2012/11/karras2012hpg_paper.pdf
	*/

	/*
	* Struct of arrays for the information of the primitives in the scene.
	*/
	struct BVHPrimitiveInfo {
		int count;
		int* primitiveNumbers;
		Bounds3f globalBounds;
		Bounds3f* bounds;
		Point3f* centroids;

		__host__ BVHPrimitiveInfo() {}
		__host__ BVHPrimitiveInfo(int primitiveN) {
			count = primitiveN;
			gpuErrCheck(cudaMalloc((void**)&primitiveNumbers, sizeof(int) * count));
			gpuErrCheck(cudaMalloc((void**)&bounds, sizeof(Bounds3f) * count));
			gpuErrCheck(cudaMalloc((void**)&centroids, sizeof(Point3f) * count));
		}

		__device__ void addPrimitive(const Primitive* primitives, int index) {
			primitiveNumbers[index] = index;
			bounds[index] = primitives[index].worldBound();
			centroids[index] = bounds[index].pMin * 0.5f + bounds[index].pMax * 0.5f;

			atomicUnion(&globalBounds, bounds[index]);
		}
	};

	__device__ inline uint32_t leftShift3(uint32_t x) {

		if (x == (1 << 10)) --x;
		x = (x | (x << 16)) & 0b00000011000000000000000011111111;
		x = (x | (x << 8)) & 0b00000011000000001111000000001111;
		x = (x | (x << 4)) & 0b00000011000011000011000011000011;
		x = (x | (x << 2)) & 0b00001001001001001001001001001001;
		return x;
	}

	/*
	* Encode a 3D position into a 30 bit morton code.
	*/
	__device__ inline uint32_t encodeToMorton(const Vector3f& v) {
		return leftShift3(v.x) | (leftShift3(v.y) << 1) | (leftShift3(v.z) << 2);
	}

	/*
	* Structure of information about the morton codes of the scene
	*/
	struct MortonPrimitives {
		int count;
		int* primitiveNumbers;
		uint32_t* mortonCodes;
		int mortonScale = 1 << 10;

		__host__ MortonPrimitives() {}
		__host__ MortonPrimitives(int primitiveN) {
			count = primitiveN;
			gpuErrCheck(cudaMalloc((void**)&primitiveNumbers, sizeof(int) * count));
			gpuErrCheck(cudaMalloc((void**)&mortonCodes, sizeof(uint32_t) * count));
		}

		__device__ void convertInfoToMorton(const BVHPrimitiveInfo* info, int index) {
			primitiveNumbers[index] = info->primitiveNumbers[index];
			mortonCodes[index] = encodeToMorton(info->globalBounds.offset(info->centroids[index]) * mortonScale);
		}
	};

	__host__ inline void sortMortonPrims(MortonPrimitives* h_morton) {

		thrust::device_ptr<int> pns = thrust::device_pointer_cast(h_morton->primitiveNumbers);
		thrust::device_ptr<uint32_t> mcs = thrust::device_pointer_cast(h_morton->mortonCodes);

		thrust::stable_sort_by_key(mcs, mcs + h_morton[0].count, pns);
	}

	struct LBVHBuildNode {
		int key;
		int visited = 0;
		LBVHBuildNode* left = nullptr;
		LBVHBuildNode* right = nullptr;
		LBVHBuildNode* parent = nullptr;
		Bounds3f bounds;

		__device__ void interiorNode(LBVHBuildNode* c1, LBVHBuildNode* c2) {
			left = c1;
			right = c2;
			left->setParent(this);
			right->setParent(this);
			visited = 0;
		}

		__device__ void setParent(LBVHBuildNode* p) {
			parent = p;
		}

		__device__ bool isLeaf() {
			return !left && !right;
		}
	};

	__device__ inline int delta(MortonPrimitives* mortons, const int index1, const int index2) {

		if ((index1 < 0 || index1 >= mortons->count) || (index2 < 0 || index2 >= mortons->count)) {
			return -1;
		}

		uint32_t differentBits = mortons->mortonCodes[index1] ^ mortons->mortonCodes[index2];

		int totalBitCount = 0;

		if (differentBits == 0) {

			totalBitCount += 40;
			differentBits = index1 ^ index2;
		}

		return __clz((int)differentBits) + totalBitCount;
	}

	__device__ inline void treeBuilding(MortonPrimitives* keys, LBVHBuildNode* leaf,
		LBVHBuildNode* interior, int i) {

		// 1. Direction of the interval
		int d = (delta(keys, i, i + 1) - delta(keys, i, i - 1));
		d = d > 0 ? 1 : -1;

		// 2. Minimum delta
		int deltaMin = delta(keys, i, i - d);

		// 3. Other extremum of interval
		int lMax = 128;

		while (delta(keys, i, i + lMax * d) > deltaMin) {
			lMax *= 4;
		}

		// Binary search of the real length
		int l = 0;
		for (int t = lMax / 2; t >= 1; t /= 2) {
			if (delta(keys, i, i + (l + t) * d) > deltaMin) {
				l += t;
			}
		}
		int j = i + l * d;

		// 4. Find split position
		int deltaNode = delta(keys, i, j);

		int s = 0;
		for (int t = ceilf(l / 2.0f); t >= 1; t = t == 1 ? 0 : ceilf(t / 2.0f)) {

			if (delta(keys, i, i + (s + t) * d) > deltaNode) {
				s += t;
			}
		}

		int gamma = i + s * d + fminf(d, 0);

		// sets nodes
		if (i < keys->count - 1) {
			interior[i].interiorNode(fminf(i, j) == gamma ? &leaf[gamma] : &interior[gamma],
				fmaxf(i, j) == gamma + 1 ? &leaf[gamma + 1] : &interior[gamma + 1]);
		}
	}

	__device__ inline void addBoundsToNode(BVHPrimitiveInfo* info, LBVHBuildNode* leaf,
		int node) {

		LBVHBuildNode* current = &leaf[node];
		current->bounds = info->bounds[current->key];

		bool newPath = true;

		do {

			current = current->parent;

			// atomic compare and swap
			int oldVal = atomicCAS(&current->visited, 0, 1);

			if (oldVal != 1) { // Was never visited, so only one bounds of its children is ready
				newPath = false;
			}
			else {

				current->bounds = bUnion(current->left->bounds, current->right->bounds);

			}

		} while (newPath && current->key != 0);
	}

	struct LBVHTraversalNode {
		int key = -1;
		LBVHTraversalNode* right = nullptr;
		LBVHTraversalNode* left = nullptr;
		Bounds3f bounds;

		__device__ bool isNull() {
			return (key == -1) && !right && !left;
		}

		__device__ bool isLeaf() {
			return key != -1 && !right && !left;
		}

	};

	class BVH {

		LBVHTraversalNode* tree;
		Primitive* primitives;

	public:

		int count;

		__host__ BVH(Primitive* d_prims, int size) {

			LBVHTraversalNode* h_tree = new LBVHTraversalNode[size];

			for (int n = 0; n < size; n++) {
				h_tree[n] = LBVHTraversalNode();
			}

			tree = passToDevice(h_tree, size);

			primitives = d_prims;
			count = size;

		}

		// Check Iterative Without using any other data structure -> https://afteracademy.com/blog/flatten-binary-tree-to-linked-list
		__device__ void setTraversalTree(LBVHBuildNode* leaf, LBVHBuildNode* interior, int node) {

			int leafOffset = ((count + 1) / 2 - 1);

			tree[node + leafOffset].key = leaf[node].key;
			tree[node + leafOffset].bounds = leaf[node].bounds;

			if (node + 1 != (count + 1) / 2) {

				tree[node].bounds = interior[node].bounds;

				int left = interior[node].left->key + (interior[node].left->isLeaf() ? leafOffset : 0);
				int right = interior[node].right->key + (interior[node].right->isLeaf() ? leafOffset : 0);

				tree[node].left = &tree[left];
				tree[node].right = &tree[right];
			}

		}

		__device__ bool simpleIntersect(const Ray& r) const {

			LBVHTraversalNode stack[64];
			int stackIndex = 0;

			const Vector3f invDir = Vector3f(1 / r.d.x, 1 / r.d.y, 1 / r.d.z);
			const int isDirNeg[3] = { invDir.x < 0, invDir.y < 0, invDir.z < 0 };

			LBVHTraversalNode current = tree[0];

			stack[stackIndex++] = LBVHTraversalNode();
			do {
				if (current.bounds.simpleIntersect(r, invDir, isDirNeg)) {

					if (current.isLeaf()) {

						if (primitives[current.key].simpleIntersect(r)) return true;
					}

					else {

						stack[stackIndex++] = *current.right;
						stack[stackIndex++] = *current.left;

					}
				}

				current = stack[--stackIndex];

			} while (!current.isNull());

			return false;
		}

		__device__ bool intersect(const Ray& r, SurfaceInteraction* it) const {

			LBVHTraversalNode stack[64];
			int stackIndex = 0;

			const Vector3f invDir = Vector3f(1 / r.d.x, 1 / r.d.y, 1 / r.d.z);
			const int isDirNeg[3] = { invDir.x < 0, invDir.y < 0, invDir.z < 0 };

			LBVHTraversalNode current = tree[0];

			stack[stackIndex++] = LBVHTraversalNode();

			bool intersection = false;

			do {
				if (current.bounds.simpleIntersect(r, invDir, isDirNeg)) {

					if (current.isLeaf()) {
						if (primitives[current.key].intersect(r, it)) intersection = true;
					}

					else {

						stack[stackIndex++] = *current.right;
						stack[stackIndex++] = *current.left;
					}
				}

				current = stack[--stackIndex];

			} while (!current.isNull());

			return intersection;
		}
	};

	__global__ void calculateMortonCodes(BVHPrimitiveInfo* info, MortonPrimitives* mortons);

	__global__ void createBuildWithoutBounds(MortonPrimitives* keys,
		LBVHBuildNode* leaf, LBVHBuildNode* interior);

	__global__ void addBoundsToBuild(BVHPrimitiveInfo* info,
		LBVHBuildNode* leaf);

	__global__ void createBVH(LBVHBuildNode* leaf, LBVHBuildNode* interior, BVH* tree);

	__host__ inline
		BVH* CreateBVHTree(Primitive* prims, BVHPrimitiveInfo* info,
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
}

#endif