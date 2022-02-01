#ifndef __image_loading_cuh__
#define __image_loading_cuh__

#include "ImageIO.cuh"
#define STB_IMAGE_IMPLEMENTATION
#include "../../extern/std_image.h"

namespace lts {

	__host__ inline void loadImageFile_f(const char* file, int* width, int* height, float* img) {

		int unused;
		img = stbi_loadf(file, width, height, &unused, 3); // load image in rgb format
		if (img == NULL) {
			printf("Couldn't load the image properly.\n");
			exit(1);
		}
	}

	__host__ inline void loadImageFile_s(const char* file, int* width, int* height, Spectrum* img) {

		int unused;
		float* image = stbi_loadf(file, width, height, &unused, 3); // load image in rgb format
		if (image == NULL) {
			printf("Couldn't load the image properly.\n");
			exit(1);
		}

		int area = (*width) * (*height);
		img = new Spectrum[area];

		for (int p = 0; p < area; p++) {
			img[p] = Spectrum(image[p * 3], image[p * 3 + 1], image[p * 3 + 2]);
		}
	}

}

#endif