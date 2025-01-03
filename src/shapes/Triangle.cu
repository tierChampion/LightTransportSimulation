﻿#ifndef __triangle_cu__
#define __triangle_cu__

#include "Triangle.cuh"

namespace lts {

	/**
	* Collection of triangles that form a 3D object.
	* @param OTW - Model matrix, defines how the object is located in the world
	* @param nTriangles - Number of triangles in the mesh
	* @param nVertices - Number of vertices in the mesh
	* @param nUVs - Number of UV coordinates in the mesh
	* @param nNormals - Number of normals in the mesh
	* @param vertexIndices - List of the indicies of the verticies
	* @param P - List of positions
	* @param N - List of normals
	* @param UV - List of UVs
	*/
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

	/**
	* Determines if the triangle intersects with the ray.
	* @param ray - Ray to test intersection with
	* @return if there is an intersection
	*/
	__device__ bool Triangle::simpleIntersect(const Ray& ray) const {

		const Point3f& p0 = mesh->p[vertices[0]];
		const Point3f& p1 = mesh->p[vertices[3]];
		const Point3f& p2 = mesh->p[vertices[6]];

		Vector3f edge1 = p1 - p0;
		Vector3f edge2 = p2 - p0;

		// Barycentric coordinates
		float u, v;

		Vector3f pVec = cross(ray.d, edge1);
		float det = dot(edge2, pVec);

		if (det == 0) return false;

		float invDet = 1.0 / det;
		Vector3f tVec = ray.o - p0;
		u = invDet * dot(tVec, pVec);

		// Barycentric u coordinate is outside of [0, 1] the triangle
		if (u < SHADOW_EPSILON || u > 1.0 + SHADOW_EPSILON) return false;

		Vector3f qVec = cross(tVec, edge2);
		v = invDet * dot(ray.d, qVec);

		// Barycentric v coordinate is outside of [0, 1] the triangle
		if (v < SHADOW_EPSILON || u + v > 1.0 + SHADOW_EPSILON) return false;

		float t = invDet * dot(edge1, qVec);

		// Ray was able to reach the object
		return t > 0 && t < ray.tMax;
	}

	/**
	* Determines if there is an intersection and if there is, compute the properties
	* of the intersection.
	* @param ray - Ray to test intersection with
	* @param tHit - Length along the ray of the intersection, if there is one
	* @param si - Properties of the interaction with the triangle
	* @return if there is an intersection
	*/
	__device__ bool Triangle::intersect(const Ray& ray, float* tHit, SurfaceInteraction* si) const {

		const Point3f& p0 = mesh->p[vertices[0]];
		const Point3f& p1 = mesh->p[vertices[1 * DATA_COUNT]];
		const Point3f& p2 = mesh->p[vertices[2 * DATA_COUNT]];

		Vector3f edge1 = p1 - p0;
		Vector3f edge2 = p2 - p0;

		// Barycentric coordinates
		float w, u, v;

		Vector3f pVec = cross(ray.d, edge2);
		float det = dot(edge1, pVec);

		if (det == 0) return false;

		float invDet = 1.0 / det;
		Vector3f tVec = ray.o - p0;
		u = invDet * dot(tVec, pVec);

		// Barycentric u coordinates is outside of [0, 1] the triangle
		if (u < SHADOW_EPSILON || u > 1.0 - SHADOW_EPSILON) return false;

		Vector3f qVec = cross(tVec, edge1);
		v = invDet * dot(ray.d, qVec);

		// Barycentric v coordinates is outside of [0, 1] the triangle
		if (v < SHADOW_EPSILON || u + v > 1.0 - SHADOW_EPSILON) return false;

		// Distance of the intersection along the ray
		float t = invDet * dot(edge2, qVec);

		// Barycentric coords are inside the triangle, INTERSECTION
		if (t > 0 && t < ray.tMax) {

			w = 1 - u - v;

			// Partial derivatives at intersection
			Vector3f dpdu, dpdv;
			Point2f uvs[3];
			getUVs(uvs);

			Vector2f duv02 = uvs[0] - uvs[2], duv12 = uvs[1] - uvs[2];
			Vector3f dp02 = p0 - p2, dp12 = p1 - p2;

			float determinant = duv02[0] * duv12[1] - duv02[1] * duv12[0];

			if (determinant == 0) {

				coordinateSystem(normalize(cross(edge2, edge1)), &dpdu, &dpdv);
			}
			else {
				float invDeterminant = 1 / determinant;
				dpdu = (dp02 * duv12[1] - dp12 * duv02[1]) * invDet;
				dpdv = (dp02 * -duv12[0] + dp12 * duv02[0]) * invDet;
			}

			// Intersection point error
			float xAbsSum = fabsf(w * p0.x) + fabsf(u * p1.x) + fabsf(v * p2.x);
			float yAbsSum = fabsf(w * p0.y) + fabsf(u * p1.y) + fabsf(v * p2.y);
			float zAbsSum = fabsf(w * p0.z) + fabsf(u * p1.z) + fabsf(v * p2.z);
			Vector3f pError = Vector3f(xAbsSum, yAbsSum, zAbsSum) * gamma(7);

			// 2D and 3D intersection points
			Point3f pHit = p0 * w + p1 * u + p2 * v;
			Point2f uvHit = uvs[0] * w + uvs[1] * u + uvs[2] * v;

			*si = SurfaceInteraction(pHit, pError, uvHit, -ray.d, dpdu, dpdv, Normal3f(0, 0, 0), Normal3f(0, 0, 0), this);

			// Setting of normals and shading values
			if (mesh->n) {

				Normal3f shadingNormal = normalize(mesh->n[vertices[0 * DATA_COUNT + NORMAL_OFFSET]] * w +
					mesh->n[vertices[1 * DATA_COUNT + NORMAL_OFFSET]] * u +
					mesh->n[vertices[2 * DATA_COUNT + NORMAL_OFFSET]] * v);
				Vector3f shadingTangent = si->dpdu;
				Vector3f shadingBitangent = cross((Vector3f)shadingNormal, shadingTangent);
				if (shadingBitangent.lengthSquared() > 0.0f) {
					shadingBitangent = normalize(shadingBitangent);
					shadingTangent = cross(shadingBitangent, (Vector3f)shadingNormal);
				}
				else {
					coordinateSystem((Vector3f)shadingNormal, &shadingTangent, &shadingBitangent);
				}

				Normal3f dndu, dndv;
				Normal3f dn1 = mesh->n[vertices[0 * DATA_COUNT + NORMAL_OFFSET]] -
					mesh->n[vertices[1 * DATA_COUNT + NORMAL_OFFSET]];
				Normal3f dn2 = mesh->n[vertices[1 * DATA_COUNT + NORMAL_OFFSET]] -
					mesh->n[vertices[2 * DATA_COUNT + NORMAL_OFFSET]];

				if (determinant == 0) {
					dndu = dndv = Normal3f(0, 0, 0);
				}
				else {
					dndu = (dn1 * duv12[1] - dn2 * duv02[1]) * invDet;
					dndv = (dn1 * -duv12[0] + dn2 * duv02[0]) * invDet;
				}

				si->setShaddingGeometry(shadingTangent, shadingBitangent, dndu, dndv, true);

				si->it.n = faceTowards(si->it.n, si->shading.n);
			}

			*tHit = t;
			return true;
		}
		return false;
	}

	/**
	* Computes the probability density function of reaching this triangle from a specific point of vue.
	* @param it - point of vue
	* @param wi - direction of the ray
	* @param PDF
	*/
	__device__ float Triangle::Pdf(const Interaction& it,
		const Vector3f& wi) const {

		// Compute the ray and the intersection if it exists
		Ray ray = it.spawnRay(wi);
		float tHit;
		SurfaceInteraction isect;
		if (!intersect(ray, &tHit, &isect)) return 0;

		// Compute Pdf
		float pdf = distanceSquared(it.p, isect.it.p) /
			(absDot(isect.it.n, -wi) * area());

		return pdf;
	}

	/**
	* Compute a uniformly distributed random point on the triangle.
	* @param u - uniform random point on a square
	* @return u uniformly mapped on the triangle
	*/
	__device__ Interaction Triangle::sample(const Point2f& u) const {

		// Map u on to the triangle in barycentric coordinates
		Point2f b = uniformSampleTriangle(u);

		const Point3f& p0 = mesh->p[vertices[0]];
		const Point3f& p1 = mesh->p[vertices[1 * DATA_COUNT]];
		const Point3f& p2 = mesh->p[vertices[2 * DATA_COUNT]];

		// Compute the point and the normal of the point
		Interaction it;
		it.p = p0 * b[0] + p1 * b[1] + p2 * (1 - b[0] - b[1]);

		if (mesh->n) {
			it.n = normalize(mesh->n[vertices[0 * DATA_COUNT + NORMAL_OFFSET]] * b[0] +
				mesh->n[vertices[1 * DATA_COUNT + NORMAL_OFFSET]] * b[1] +
				mesh->n[vertices[2 * DATA_COUNT + NORMAL_OFFSET]] * (1 - b[0] - b[1]));
		}
		else {
			it.n = normalize(Normal3f(cross(p1 - p0, p2 - p0)));
		}

		// Compute the error of the generated point
		Point3f pAbsSum = abs(p0 * b[0]) + abs(p1 * b[1]) + abs(p2 * (1 - b[0] - b[1]));
		it.pError = Vector3f(pAbsSum) * gamma(6);

		return it;
	}
}

#endif