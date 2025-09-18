# Zig + CUDA Raytracer

Implementing a path tracer in CUDA using the C CUDA Runtime API and glueing it together with Zig's great C interop.

Requirements:
- Raylib 5.0 in `external/raylib-5.5_linux_amd64/`
- stb_image_write in `external/stb_image_write.h`
- CUDA 13.0 (only tested this one)

## Current progress:
Spheres:

![alt text](https://github.com/chrarvi/zray/blob/main/assets/render.png?raw=true)

Teapot: (No BVH yet so super inefficient with the number of tris)

![alt text](https://github.com/chrarvi/zray/blob/main/assets/teapot.png?raw=true)

**References**:
- https://raytracing.github.io/books/RayTracingInOneWeekend.html
