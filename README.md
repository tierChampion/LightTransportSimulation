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

For the time being, only a scene with no subject inside of it can be rendered via the custom .scene file format which must be selected in the main via the SCENE_FILE constant.
The possible materials applied to a scene are also very limited for now.
To add custom assets, 3D models must be in the .obj format, but after a first use, a .bin file will be created for faster reuse, while images must be in a resolution that is a power of 2.

# Future

- Making the scene loading process more flexible by allowing for all possible materials, lights and adding outside meshes.
- Adding a few other things from PBRT like the infinite light, bump maps and maybe other rendering algorithms (probably not happening because of memory limitations).
- Simplifying the process of making a scene. 
