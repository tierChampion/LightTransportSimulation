#ifndef __fileio_cuh__
#define __fileio_cuh__

#include <iostream>
#include <fstream>
#include <string>
#include <string_view>
#include <charconv>

#include "../shapes/Triangle.cuh"

namespace lts {

	/**
	* Write the mesh data to a binary file for faster loading in future rendering.
	* @param meshName - name of the mesh to save. The final file will be meshName.bin
	* @param nTri, nP, nUV, nNorm - size of each data of the mesh
	* @param p, uv, n, indx - data of the mesh
	*/
	__host__ inline void binMeshWrite(std::string meshName, int nTri, int nP, int nUV, int nNorm,
		Point3f* p, Point2f* uv, Normal3f* n, int* indx) {

		std::fstream bin;
		bin.open(meshName + ".bin", std::fstream::out | std::fstream::binary);
		assert(bin);

		// Number of triangles
		bin.write((char*)&nTri, sizeof(int));
		// Number of points
		bin.write((char*)&nP, sizeof(int));
		// Number of uvs
		bin.write((char*)&nUV, sizeof(int));
		// Number of normals
		bin.write((char*)&nNorm, sizeof(int));
		// Points
		bin.write((char*)p, nP * sizeof(Point3f));
		// UVs
		bin.write((char*)uv, nUV * sizeof(Point2f));
		// Normals
		bin.write((char*)n, nNorm * sizeof(Normal3f));
		// Indices
		bin.write((char*)indx, (9 * nTri) * sizeof(int));

		bin.close();
	}

	/**
	* Splits the given string_view along the character del
	* @param s - string_view to split
	* @param del - split character
	*/
	__host__ inline std::vector<std::string_view> split(std::string_view s, char del) {

		std::vector<std::string_view> result;
		result.reserve(4);
		int start = 0;
		int end = s.find(del);

		while (end != -1) {
			result.emplace_back(s.substr(start, end - start));
			start = end + 1;
			end = s.find(del, start);
		}

		result.emplace_back(s.substr(start, end - start));
		return result;
	}

	/**
	* Read and load the obj file of the given mesh name.
	* @param meshName - name of the mesh to load. Read file is meshName.obj
	*/
	__host__ inline TriangleMesh* parseMeshFile(std::string meshName, const Transform& OTW) {

		int nTriangles = 0;
		std::vector<int> h_indices(0);
		std::vector<Point3f> h_p(0);
		std::vector<Point2f> h_uv(0);
		std::vector<Normal3f> h_n(0);
		bool hasAllData = false;

		// open file
		std::ifstream obj(meshName + ".obj");
		std::string token;

		if (obj.is_open()) {
			while (obj >> token) {

				if (token == "v") {
					Point3f p;
					obj >> token; p.x = std::stof(token);
					obj >> token; p.y = std::stof(token);
					obj >> token; p.z = std::stof(token);

					h_p.emplace_back(p);
				}
				else if (token == "vt") {
					Point2f uv;
					obj >> token; uv.x = std::stof(token);
					obj >> token; uv.y = std::stof(token);

					h_uv.emplace_back(uv);
				}
				else if (token == "vn") {
					Normal3f n;
					obj >> token; n.x = std::stof(token);
					obj >> token; n.y = std::stof(token);
					obj >> token; n.z = std::stof(token);

					h_n.emplace_back(n);

				}
				else if (token == "f") {

					if (!hasAllData) {
						hasAllData = true;
						h_indices.reserve(3 * h_p.size());
					}

					for (int v = 1; v <= 3; v++) {

						obj >> token;

						std::vector<std::string_view> vertexData = split(std::string_view(token), '/');

						int i;
						std::from_chars(vertexData[0].data(),
							vertexData[0].data() + vertexData[0].size(), i);
						h_indices.emplace_back(i - 1); // p
						if (h_uv.size() != 0) {
							std::from_chars(vertexData[1].data(),
								vertexData[1].data() + vertexData[1].size(), i);
							h_indices.emplace_back(i - 1); // uv
						}
						else {
							h_indices.emplace_back(0);
						}

						std::from_chars(vertexData[2].data(),
							vertexData[2].data() + vertexData[2].size(), i);
						h_indices.emplace_back(i - 1); // n
					}
					nTriangles++;
				}
			}
		}
		else {
			std::cout << "Error: Unable to open file " << meshName << "!" << std::endl;
		}

		obj.close();

		binMeshWrite(meshName, nTriangles, h_p.size(), h_uv.size(), h_n.size(),
			h_p.data(), h_uv.data(), h_n.data(), h_indices.data());

		TriangleMesh* mesh = new TriangleMesh(OTW, nTriangles, h_p.size(), h_uv.size(), h_n.size(),
			h_indices.data(), h_p.data(),
			h_n.data(),
			h_uv.size() == 0 ? nullptr : h_uv.data());

		std::cout << "Saved the mesh to a corresponding binary file." << std::endl;

		return mesh;
	}

	/**
	* Read and load the given binary file of a mesh.
	* @param bin - opened binary file containing the data of a triangle mesh
	*/
	__host__ inline TriangleMesh* binMeshRead(std::ifstream& bin, const Transform& OTW) {

		int nTri, nP, nUV, nNorm;


		// Number of triangles
		bin.read((char*)&nTri, sizeof(int));
		// Number of points
		bin.read((char*)&nP, sizeof(int));
		// Number of uvs
		bin.read((char*)&nUV, sizeof(int));
		// Number of normals
		bin.read((char*)&nNorm, sizeof(int));

		int* indx = new int[9 * nTri];
		Point3f* p = new Point3f[nP];
		Point2f* uv = new Point2f[nUV];
		Normal3f* n = new Normal3f[nNorm];

		// Points
		bin.read((char*)p, nP * sizeof(Point3f));
		// UVs
		bin.read((char*)uv, nUV * sizeof(Point2f));
		// Normals
		bin.read((char*)n, nNorm * sizeof(Normal3f));
		// Indices
		bin.read((char*)indx, 9 * nTri * sizeof(int));

		bin.close();

		TriangleMesh* mesh = new TriangleMesh(OTW, nTri, nP, nUV, nNorm,
			indx, p,
			n,
			nUV == 0 ? nullptr : uv);

		return mesh;
	}

	/**
	* Load a triangle mesh with the given. Will either load it from an obj file or a binary file, if available.
	* @param meshName - name of the mesh to load. The loaded file will be meshName.bin if it exist
	* or else it will be meshName.obj
	*/
	__host__ inline TriangleMesh* parseMesh(std::string meshName, const Transform& OTW) {

		std::ifstream stream;
		stream.open(meshName + ".bin", std::fstream::in | std::fstream::binary);

		TriangleMesh* mesh;

		if (!stream.good()) {
			mesh = parseMeshFile(meshName, OTW);
		}
		else {
			mesh = binMeshRead(stream, OTW);
		}

		std::cout << "(M) Loaded mesh: " << meshName << "\n" <<
			"	mesh complexity: " << mesh->nTriangles << " triangles." << std::endl;

		return mesh;
	}


	__host__ inline void readApplicationParameters(std::string parameterFile,
		int* width, int* height, int* spp, int* bounce, int* roulette, int* format,
		std::string* output, std::string* scene, std::string* subject) {

		std::ifstream stream;
		stream.open(parameterFile);

		std::string token;

		if (stream.is_open()) {

			stream >> token; *width = std::stoi(token);
			stream >> token; *height = std::stoi(token);
			stream >> token; *spp = std::stoi(token);
			stream >> token; *bounce = std::stoi(token);
			stream >> token; *roulette = std::stoi(token);
			stream >> token; *format = std::stoi(token);
			stream >> token; *output = token + (*format ? ".pfm" : ".ppm");
			stream >> token; *scene = token;
			stream >> token; *subject = token;
		}

		stream.close();
	}
}

#endif