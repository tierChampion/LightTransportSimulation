#ifndef __image_input_cuh__
#define __image_input_cuh__

#include "GeneralHelper.cuh"
#include "../rendering/Spectrum.cuh"
#define STB_IMAGE_IMPLEMENTATION
#include "../../extern/stb_image.h"

namespace lts {

	__host__ inline Spectrum* loadImageFile_s(const char* file, int* width, int* height) {

		int unused;
		float* image = stbi_loadf(file, width, height, &unused, 3); // load image in rgb format
		if (image == NULL) {
			printf("Couldn't load the image properly.\n");
			exit(1);
		}

		int area = (*width) * (*height);
		Spectrum* img = new Spectrum[area];

		int offset = 0;
		int index = 0;

		for (int h = *height - 1; h >= 0; h--) {

			offset = h * (*width);

			for (int w = 0; w < *width; w++) {
				int p = offset + w;
				img[index++] = Spectrum(image[p * 3], image[p * 3 + 1], image[p * 3 + 2]);
			}
		}

		return img;
	}
}

#endif