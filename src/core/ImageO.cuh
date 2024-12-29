#ifndef __image_output_cuh__
#define __image_output_cuh__

#include "GeneralHelper.cuh"
#include "../rendering/Spectrum.cuh"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../extern/stb_image_write.h"

namespace lts {

	const int n = 1;
	const static bool hostLittleEndian = (*(char*)&n == 1);

#define GAMMA 1 / 2.2f

	__host__ inline bool writeToPFM(const char* filename, const float* rgb, int width, int height) {

		FILE* fp;
		float scale;

		fp = fopen(filename, "wb");
		if (!fp) {
			fprintf(stderr, "Error: Unable to open output PFM file %s.\n", filename);
			return false;
		}

		std::unique_ptr<float[]> scanline(new float[3 * width]);

		if (fprintf(fp, "PF\n") < 0) goto fail;

		if (fprintf(fp, "%d %d\n", width, height) < 0) goto fail;

		scale = hostLittleEndian ? -1.0f : 1.0f;
		if (fprintf(fp, "%f\n", scale) < 0) goto fail;

		for (int y = 0; y < height; y++) {
			for (int x = 0; x < 3 * width; x++) {
				scanline[x] = rgb[y * width * 3 + x];
			}
			if (fwrite(&scanline[0], sizeof(float), width * 3, fp) < (size_t)(width * 3)) goto fail;
		}

		fclose(fp);
		return true;

	fail:
		fprintf(stderr, "Error while writing to output PFM file %s.\n", filename);
		fclose(fp);
		return false;
	}

	__host__ inline int toRGB(float x) { return int(powf(clamp(x), GAMMA) * 255 + 0.5); }

	__host__ inline bool writeToPPM(const char* filename, const float* rgb, int width, int height) {

		FILE* fp;

		fp = fopen(filename, "w");
		if (!fp) {
			fprintf(stderr, "Error: Unable to open output PPM file %s.\n", filename);
			return false;
		}

		if (fprintf(fp, "P3\n") < 0) goto fail;

		if (fprintf(fp, "%d %d\n%d\n", width, height, 255) < 0) goto fail;

		for (int i = 0; i < width * height; i++) {
			fprintf(fp, "%d %d %d ", toRGB(rgb[3 * i]), toRGB(rgb[3 * i + 1]), toRGB(rgb[3 * i + 2]));
		}

		fclose(fp);
		return true;

	fail:
		fprintf(stderr, "Error while writing to output PPM file %s. \n", filename);
		return false;
	}

	__host__ bool outputFile(const char* filename, const float* rgb, int width, int height, int format) {

		// TODO: PNG cant take float data 

		int result = stbi_write_hdr(filename, width, height, 3, rgb);

		if (result == 0) {
			std::cerr << "ERROR: Problem when trying to output the image file." << std::endl;
			return false;
		}

		return true;
	}
}

#endif