# Light transport simulation

Project based on the works of [PBRT3](https://www.pbr-book.org/3ed-2018/contents). Instead of using normal multithreading like in PBRT,
this project uses CUDA to parallelise the rendering.

## Implemented sections

All geometries except animated transforms, triangle meshes, parallel bounding volume hierarchies, radiometry, perspective camera, stratified and uniform sampling, filters, films, lambertian reflection, specular reflection and transmission, microfacet models, bsdfs, a few materials, procedural and image textures, point and area lights.
The algorithm used for rendering is the standard path tracing algorithm described in section 14.
The parallel BVH acceleration data structure is based on the pseudo-code in this [paper](https://developer.nvidia.com/blog/parallelforall/wp-content/uploads/2012/11/karras2012hpg_paper.pdf) from NVIDIA released in 2012.

## Usability

This project was only tested on a GTX 1650ti with compute capability 7.5 and CUDA 11.2. Other graphic cards might not react the same way to the code and crash. Also, since a lot of memory is allocated on the GPU during runtime, very large renders might not be possible.

# How to use

To start off, the rendering parameters are defined in the app.params file (add link). In there, are listed in order: width and height of the render, samples per pixel of the render, maximum light bounces, first bounce where russian roulette is applied, format of the render (either 0 for ppm or 1 for pfm), scene file to use and subject file to use (no extensions).

In order to create a custom scene, a .scene file must be created. The scene file contains the following information: parameters of the camera (position, point to look at, up direction, vertical field of view and distance to the focus point), transform of the subject (position, axis of rotation, angle in radians of rotation and scale), number of textures and texture files if needed, number of meshes and mesh files if needed (no extensions), number of materials and material definitions if needed, number of light meshes with their luminance if needed and if there is an infinite light with its luminance if needed.

Finally, the .subject file resembles the scene file a lot. It is formatted just like the scene file but without the camera information, the transform and the lights information.

For the material definitions, there are three options for the moment either a MatteMaterial (L), a GlassMaterial (G) or a MirrorMaterial (R). Each of these materials require textures, which can either be Constant (c), Image (i), FBM (f), Wrinkled (w) or Marble (m). The constant texture requires an rgb color, the image texture requires the index of the image used, the FBM and windy textures require a whole number of octaves and an omega value. Finally, each material requires either a roughness value (M and R) or a index of refraction (G).

For all of these files, all that is required is that the different information be seperated by white spaces. Anything after the end of the data will not be read and if an setting is not used, no parameter must be defined.

# Future

- Adding the metal material with predifined parameters (gold, copper, bronze, etc.)
- Adding a few other things from PBRT like the bump maps and maybe other rendering algorithms (probably not happening because of memory limitations).

