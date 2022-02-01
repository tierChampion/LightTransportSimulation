#ifndef __triangle_cu__
#define __triangle_cu__

#include "Triangle.cuh"

namespace lts {

	__host__ TriangleMesh::TriangleMesh(const Transform& OTW, int nTriangles, int nVertices, int nUVs, int nNormals,
		const int* vertexIndices, const Point3f* P, const Normal3f* N, const Point2f* UV) :
		nTriangles(nTriangles)
	{
		this->vertexIndices = passToDevice(vertexIndices, 9 * nTriangles);

		Point3f* worldP = new Point3f[nVertices];
		for (int i = 0; i < nVertices; i++) {
			worldP[i] = OTW(P[i]);
		}
		this->p = passToDevice(worldP, nVertices);

		if (UV) {
			this->uv = passToDevice(UV, nUVs);
		}

		if (N) {
			Normal3f* worldN = new Normal3f[nNormals];
			for (int i = 0; i < nNormals; i++) {
				worldN[i] = OTW(N[i]);
			}
			this->n = passToDevice(worldN, nNormals);
		}
	}
}

#endif