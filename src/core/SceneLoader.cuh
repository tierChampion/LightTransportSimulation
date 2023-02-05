#ifndef __scene_loader_cuh__
#define __scene_loader_cuh__

#include "FileIO.cuh"
#include "../acceleration/BVH.cuh"
#include "../acceleration/Scene.cuh"
#include "../materials/textures/Procedural.cuh"
#include "../materials/textures/ImageTexture.cuh"
#include "../rendering/Camera.cuh"

namespace lts {

	__global__ void materialInitKernel(Material** materials,
		char* materialTypes, char* textureTypes, int* matStarts, int* texStarts,
		float* materialParams, float* extras,
		Mipmap* mpmps, int materialCount);

	__global__ void sceneInitKernel(Triangle* tris, Primitive* prims,
		BVHPrimitiveInfo* info, MortonPrimitives* morton, TriangleMesh* meshes,
		Material** materials, Light** lights, float* LEs,
		int meshCount, int materialCount, int areaLightCount);

	__global__ void infiniteLightsKernel(BVHPrimitiveInfo* info, Light** lights,
		float* LEs, TriangleMesh* meshes, int areaLightCount, int meshCount);

	__host__ inline void readFromCorrectFile(std::ifstream& s1, std::ifstream& s2,
		bool conditionFor1, std::string* token) {

		if (conditionFor1) s1 >> *token;
		else s2 >> *token;
	}

	__host__ inline void readCameraInformation(Camera** camera,
		float aperture, int width, int height, Filter* filter,
		std::string token, std::ifstream& sceneStream) {

		float lfx, lfy, lfz;
		float lax, lay, laz;
		float upx, upy, upz;
		float vfov, dist;
		sceneStream >> token; lfx = std::stof(token);
		sceneStream >> token; lfy = std::stof(token);
		sceneStream >> token; lfz = std::stof(token);

		sceneStream >> token; lax = std::stof(token);
		sceneStream >> token; lay = std::stof(token);
		sceneStream >> token; laz = std::stof(token);

		sceneStream >> token; upx = std::stof(token);
		sceneStream >> token; upy = std::stof(token);
		sceneStream >> token; upz = std::stof(token);

		sceneStream >> token; vfov = std::stof(token);
		sceneStream >> token; dist = std::stof(token);
		*camera = new Camera(Point3f(lfx, lfy, lfz), Point3f(lax, lay, laz),
			Vector3f(upx, upy, upz), vfov, aperture, dist, width, height, filter);
	}

	__host__ inline void readTransformInformation(Transform** STW,
		std::string token, std::ifstream& sceneStream) {

		float tx, ty, tz;
		float ax, ay, az, theta;
		float scl;
		sceneStream >> token; tx = std::stof(token);
		sceneStream >> token; ty = std::stof(token);
		sceneStream >> token; tz = std::stof(token);

		sceneStream >> token; ax = std::stof(token);
		sceneStream >> token; ay = std::stof(token);
		sceneStream >> token; az = std::stof(token);
		sceneStream >> token; theta = std::stof(token);

		sceneStream >> token; scl = std::stof(token);

		*STW = new Transform(Vector3f(tx, ty, tz), Vector3f(ax, ay, az), theta, scl);
	}

	__host__ inline void readImageInformation(int* imgCount, int* subImgCount, std::vector<Mipmap>& mpmps,
		std::string token, std::ifstream& sceneStream, std::ifstream& subjectStream, bool hasSubject, const std::string absPath) {
		// Number of images or mipmaps
		if (hasSubject) {
			subjectStream >> token; *subImgCount = std::stoi(token);
			*imgCount = *subImgCount;
		}
		sceneStream >> token; *imgCount += std::stoi(token);

		// Mipmaps creation
		for (int i = 0; i < *imgCount; i++) {
			bool useSceneStream = (i >= *subImgCount);

			readFromCorrectFile(sceneStream, subjectStream, useSceneStream, &token);
			ImageWrap wrapMode = (ImageWrap)clamp(std::stoi(token), 0, 2);
			readFromCorrectFile(sceneStream, subjectStream, useSceneStream, &token);
			mpmps.emplace_back(CreateMipMap((absPath + token).c_str(), wrapMode));
		}
	}

	__host__ inline void readMeshInformation(int* meshCount, int* subjectMeshCount, std::vector<std::string>& meshFiles,
		std::string token, std::ifstream& sceneStream, std::ifstream& subjectStream, bool hasSubject, const std::string absPath) {
		if (hasSubject) {
			subjectStream >> token; *subjectMeshCount = std::stoi(token);
			*meshCount = *subjectMeshCount;
		}
		sceneStream >> token; *meshCount += std::stoi(token);
		// Mesh files
		for (int m = 0; m < *meshCount; m++) {
			bool useSceneStream = (m >= *subjectMeshCount);

			readFromCorrectFile(sceneStream, subjectStream, useSceneStream, &token);
			meshFiles.emplace_back(absPath + token);
		}
	}

	__host__ inline void readMaterialInformation(int* materialCount, int* subjectMaterialCount,
		int* matParamCounter, int* texParamCounter, int subjectImgCount,
		std::vector<char>& matTypes, std::vector<int>& matStarts,
		std::vector<char>& texTypes, std::vector<int>& texStarts,
		std::vector<float>& matParams, std::vector<float>& matExtras,
		std::string token, std::ifstream& sceneStream, std::ifstream& subjectStream, bool hasSubject) {

		if (hasSubject) {
			subjectStream >> token; *subjectMaterialCount = std::stoi(token);
			*materialCount = *subjectMaterialCount;
		}
		sceneStream >> token; *materialCount += std::stoi(token);
		// Parameters
		for (int mat = 0; mat < *materialCount; mat++) {
			bool useSceneStream = (mat >= *subjectMaterialCount);

			readFromCorrectFile(sceneStream, subjectStream, useSceneStream, &token);
			matTypes.emplace_back(token[0]);
			// Material settings
			int matLength = 0;
			char matType = matTypes.back();
			if (matType == 'G') matLength = 2;
			else if (matType != 'M') matLength = 1;
			matStarts.emplace_back(*matParamCounter);
			*matParamCounter += matLength;
			// Texture settings
			for (int t = 0; t < matLength; t++) {
				readFromCorrectFile(sceneStream, subjectStream, useSceneStream, &token);
				texTypes.emplace_back(token[0]);
				int texLength = 0;
				char texType = texTypes.back();

				if (texType == 'i') texLength = 1; // image
				else if (texType == 'f' || texType == 'w') texLength = 2; // fbm or windy
				else if (texType == 'c') texLength = 3; // constant
				else if (texType == 'm') texLength = 4; // marble

				for (int i = 0; i < texLength; i++) {
					if (useSceneStream) {
						sceneStream >> token;
						if (texType == 'i')
							matParams.emplace_back(std::stof(token) + subjectImgCount);
						else
							matParams.emplace_back(std::stof(token));
					}
					else {
						subjectStream >> token; matParams.emplace_back(std::stof(token));
					}
				}
				texStarts.emplace_back(*texParamCounter);
				*texParamCounter += texLength;
			}
			// Extra parameter (roughness or eta)
			readFromCorrectFile(sceneStream, subjectStream, useSceneStream, &token);
			matExtras.emplace_back(std::stof(token));
		}
	}

	__host__ inline bool readLightInformation(int* areaLightMeshCount,
		std::vector<float>& LEs,
		std::string token, std::ifstream& sceneStream) {
		// Number of meshes with light
		sceneStream >> token; *areaLightMeshCount = std::stoi(token);
		// AreaLight parameters
		for (int al = 0; al < *areaLightMeshCount; al++) {
			sceneStream >> token; LEs.emplace_back(std::stof(token));
			sceneStream >> token; LEs.emplace_back(std::stof(token));
			sceneStream >> token; LEs.emplace_back(std::stof(token));
		}
		// InfiniteLight, if necessary
		sceneStream >> token; bool hasInfinite = std::stoi(token) > 0;
		if (hasInfinite) {
			sceneStream >> token; LEs.emplace_back(std::stof(token));
			sceneStream >> token; LEs.emplace_back(std::stof(token));
			sceneStream >> token; LEs.emplace_back(std::stof(token));
		}

		return hasInfinite;
	}

	__host__ inline Scene* parseScene(Camera** camera, Filter* filter,
		float aperture, int filmWidth, int filmHeight,
		const std::string absPath, std::string sceneName, std::string subjectName = "") {

		// Only support arealights
		// No lights in the subject file
		// No metal material (to add with special functions for particular metals)
		// Materials arent reused for lower memory usage if they already exist
		// Separate into multiple functions for readability

		int imgCount = 0;
		std::vector<Mipmap> mpmps;
		Transform* subjectTransform;
		int meshCount = 0;
		int subjectMeshCount = 0;
		std::vector<std::string> meshFiles;
		int materialCount = 0;
		std::vector<char> matTypes;
		std::vector<char> texTypes;
		std::vector<float> matParams;
		std::vector<int> texStarts;
		std::vector<int> matStarts;
		std::vector<float> matExtras;
		int areaLightMeshCount = 0;
		bool hasInfinite = false;
		std::vector<float> LEs;
		int matParamCounter = 0;
		int texParamCounter = 0;

		bool hasSubject = subjectName != "";

		std::string token;
		std::ifstream sceneStream;
		std::ifstream subjectStream;
		sceneStream.open(absPath + sceneName + ".scene");
		if (hasSubject) subjectStream.open(absPath + subjectName + ".subject");

		/* SCENE FILE PARSING */
		if (sceneStream.is_open() && (hasSubject == subjectStream.is_open())) {
			/* Camera */
			readCameraInformation(camera, aperture, filmWidth, filmHeight, filter, token, sceneStream);
			/* Subject Transform */
			readTransformInformation(&subjectTransform, token, sceneStream);
			/* Images */
			int subjectImgCount = 0;
			readImageInformation(&imgCount, &subjectImgCount, mpmps,
				token, sceneStream, subjectStream, hasSubject, absPath);
			/* Meshes */
			readMeshInformation(&meshCount, &subjectMeshCount, meshFiles,
				token, sceneStream, subjectStream, hasSubject, absPath);
			/* Materials */
			int subjectMaterialCount = 0;
			readMaterialInformation(&materialCount, &subjectMaterialCount,
				&matParamCounter, &texParamCounter, subjectImgCount,
				matTypes, matStarts, texTypes, texStarts, matParams, matExtras,
				token, sceneStream, subjectStream, hasSubject);
			/* Lights */
			hasInfinite = readLightInformation(&areaLightMeshCount, LEs, token, sceneStream);
		}
		sceneStream.close();
		if (hasSubject) subjectStream.close();

		/* Meshes initialisation */

		std::vector<TriangleMesh> h_meshes;
		int triCount = 0;
		int highestTriCount = 0;
		int lightCount = 0;

		for (int f = 0; f < meshCount; f++) {

			std::string file = meshFiles[f];

			Transform t;

			if (f >= subjectMeshCount) {
				t = Transform(1);
			}
			else {
				t = *subjectTransform;
			}

			h_meshes.push_back(*parseMesh(file, t));

			triCount += h_meshes.back().nTriangles;
			if (f >= meshCount - areaLightMeshCount) lightCount += h_meshes.back().nTriangles;
			if (h_meshes.back().nTriangles > highestTriCount) highestTriCount = h_meshes.back().nTriangles;
		}
		// Add the infinite light if necessary
		lightCount += hasInfinite;

		TriangleMesh* d_meshes = passToDevice(h_meshes.data(), meshCount);

		/* Materials initialisation */
		Material** d_materials;
		Mipmap* d_mpmps = passToDevice(mpmps.data(), imgCount);
		char* d_matTypes = passToDevice(matTypes.data(), materialCount);
		char* d_texTypes = passToDevice(texTypes.data(), matParamCounter);
		int* d_matStarts = passToDevice(matStarts.data(), materialCount);
		int* d_texStarts = passToDevice(texStarts.data(), matParamCounter);
		float* d_matParams = passToDevice(matParams.data(), texParamCounter);
		float* d_extras = passToDevice(matExtras.data(), materialCount);
		gpuErrCheck(cudaMalloc(&d_materials, sizeof(Material**))); // allocate right size

		// Material kernel
		materialInitKernel << <1, materialCount >> > (d_materials,
			d_matTypes, d_texTypes, d_matStarts, d_texStarts, d_matParams, d_extras,
			d_mpmps, materialCount);
		gpuErrCheck(cudaDeviceSynchronize());
		gpuErrCheck(cudaPeekAtLastError());

		/* Primitives & Acceleration Structure allocation and initialisation */

		// Acceleration structure
		BVHPrimitiveInfo* h_info = new BVHPrimitiveInfo(triCount);
		BVHPrimitiveInfo* d_info = passToDevice(h_info);
		MortonPrimitives* h_morton = new MortonPrimitives(triCount);
		MortonPrimitives* d_morton = passToDevice(h_morton);
		// Primitives
		Triangle* d_tris;
		Primitive* d_prims;
		Light** d_lights;
		float* d_les = passToDevice(LEs.data(), 3 * (areaLightMeshCount + hasInfinite));
		gpuErrCheck(cudaMalloc(&d_tris, (triCount) * sizeof(Triangle)));
		gpuErrCheck(cudaMalloc(&d_prims, (triCount) * sizeof(Primitive)));

		gpuErrCheck(cudaMalloc(&d_lights, sizeof(Light**)));

		dim3 block(BLOCK_SIZE, 1);
		dim3 grid(highestTriCount / BLOCK_SIZE +
			(highestTriCount % BLOCK_SIZE != 0), meshCount);

		// scene init kernel
		sceneInitKernel << <grid, block >> > (d_tris, d_prims, d_info, d_morton,
			d_meshes, d_materials, d_lights, d_les, meshCount, materialCount, areaLightMeshCount);
		gpuErrCheck(cudaDeviceSynchronize());
		gpuErrCheck(cudaPeekAtLastError());
		if (hasInfinite) {
			infiniteLightsKernel << <1, 1 >> > (d_info, d_lights, d_les, d_meshes,
				areaLightMeshCount, meshCount);
			gpuErrCheck(cudaDeviceSynchronize());
			gpuErrCheck(cudaPeekAtLastError());
		}
		// morton kernel
		calculateMortonCodes << <grid, block >> > (d_info, d_morton);
		gpuErrCheck(cudaDeviceSynchronize());
		gpuErrCheck(cudaPeekAtLastError());

		BVH* d_tree = CreateBVHTree(d_prims, d_info, h_morton, d_morton, triCount);
		// Deallocation
		cudaFree(d_info);
		delete h_info;
		cudaFree(d_morton);
		delete h_morton;
		cudaFree(d_matTypes);
		cudaFree(d_texTypes);
		cudaFree(d_matStarts);
		cudaFree(d_texStarts);
		cudaFree(d_matParams);
		cudaFree(d_extras);
		cudaFree(d_les);

		Scene* scene = new Scene(d_tree, d_lights, meshCount, lightCount);

		return passToDevice(scene);
	}

	__device__ inline void setMatTexture(Texture<Spectrum>** texture, Mipmap* mpmps,
		char* textureTypes, float* materialParams,
		int matStart, int start) {

		switch (textureTypes[matStart]) {
		case 'c': *texture = new ConstantTexture(Spectrum(materialParams[start],
			materialParams[start + 1],
			materialParams[start + 2])); break;
		case 'b': *texture = new BilerpTexture(new UVMapping2D(1, 1, 0, 0),
			Spectrum(1.00f, 0.01f, 0.01f),
			Spectrum(0.01f, 1.0f, 0.01f),
			Spectrum(0.01f, 0.0f, 1.01f),
			Spectrum(0.01f, 0.01f, 0.0f)); break;
		case 'i': *texture = new ImageTexture(new UVMapping2D(1, 1, 0, 0), &mpmps[(int)materialParams[start]]); break;
		case 'f': *texture = new FBmTexture<Spectrum>(Transform(),
			materialParams[start],
			materialParams[start + 1]); break;
		case 'w': *texture = new WrinkledTexture<Spectrum>(Transform(),
			materialParams[start],
			materialParams[start + 1]); break;
		case 'v': *texture = new WindyTexture<Spectrum>(Transform()); break;
		case 'm': *texture = new MarbleTexture(Transform(),
			materialParams[start],
			materialParams[start + 1],
			materialParams[start + 2],
			materialParams[start + 3]); break;
		default: printf("Error while reading the texture type in scene loading. Found %c at %i\n",
			textureTypes[matStart], matStart);
		}
	}
}

#endif