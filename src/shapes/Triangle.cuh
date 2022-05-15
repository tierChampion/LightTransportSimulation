#ifndef __triangle_cuh__
#define __triangle_cuh__

#include "../geometry/Transform.cuh"
#include "../sampling/Sampling.cuh"

namespace lts {

	const static int DATA_COUNT = 3;
	const static int UV_OFFSET = 1;
	const static int NORMAL_OFFSET = 2;

	struct TriangleMesh {

		const int nTriangles;
		const int* vertexIndices = nullptr; // format: p(x), uv(x), n(x), p(y), uv(y), n(y), p(z), uv(z), n(z)
		Point3f* p = nullptr; // point, one per vertex
		Normal3f* n = nullptr; // normal, one per vertex
		Point2f* uv = nullptr; // uv mapping coordinate, three per triangle (change for the same vertex)

		__host__ TriangleMesh(const Transform& OTW, int nTriangles, int nVertices, int nUVs, int nNormals,
			const int* vertexIndices, const Point3f* P, const Normal3f* N, const Point2f* UV);
	};

	class Triangle {

		TriangleMesh* mesh;
		const int* vertices;

	public:

		__device__ Triangle(TriangleMesh* mesh, int triNumber) :
			mesh(mesh) {
			vertices = &mesh->vertexIndices[9 * triNumber];
		}

		__device__ bool simpleIntersect(const Ray& ray) const;

		__device__ bool intersect(const Ray& ray, float* tHit, SurfaceInteraction* si) const;

		__device__ Bounds3f worldBound() const {
			const Point3f& p0 = mesh->p[vertices[0 * DATA_COUNT]];
			const Point3f& p1 = mesh->p[vertices[1 * DATA_COUNT]];
			const Point3f& p2 = mesh->p[vertices[2 * DATA_COUNT]];
			return bUnion(Bounds3f(p0, p1), p2);
		}

		__device__ float area() const {
			const Point3f& p0 = mesh->p[vertices[0]];
			const Point3f& p1 = mesh->p[vertices[1 * DATA_COUNT]];
			const Point3f& p2 = mesh->p[vertices[2 * DATA_COUNT]];
			return 0.5f * cross(p1 - p0, p2 - p0).length();
		}

		__device__ float Pdf(const Interaction& it) const {
			return 1 / area();
		}

		__device__ float Pdf(const Interaction& it,
			const Vector3f& wi) const;

		__device__ Interaction sample(const Point2f& u) const;

		__device__ Interaction sample(const Interaction& it, const Point2f& u) const {

			return sample(u);
		}

	private:

		__device__ void getUVs(Point2f uv[3]) const {
			if (mesh->uv) {
				uv[0] = mesh->uv[vertices[0 * DATA_COUNT + UV_OFFSET]];
				uv[1] = mesh->uv[vertices[1 * DATA_COUNT + UV_OFFSET]];
				uv[2] = mesh->uv[vertices[2 * DATA_COUNT + UV_OFFSET]];
			}
			else {
				uv[0] = Point2f(0, 0);
				uv[1] = Point2f(1, 0);
				uv[2] = Point2f(1, 1);
			}
		}
	};
}

#endif