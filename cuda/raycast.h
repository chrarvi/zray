// in lib.h

#ifndef RAYCAST_H_
#define RAYCAST_H_

#include <stdlib.h>
#include <stdbool.h>

#if defined(__cplusplus)
    #define EXTERN_C extern "C"
#else
    #define EXTERN_C extern
#endif

typedef float mat4[4][4];
typedef struct {
    float x;
    float y;
    float z;
} vec3;

typedef struct {
    float x;
    float y;
    float z;
    float w;
} vec4;

typedef struct {
    vec3 min;
    vec3 max;
} AABB;

typedef struct {
    unsigned int image_width;
    unsigned int image_height;
    float focal_length;
    unsigned int samples_per_pixel;
    int max_depth;
    mat4 camera_to_world;
    mat4 inv_proj;
    bool temporal_averaging;
} CameraData;

typedef enum {
    MAT_LAMBERTIAN = 0,
    MAT_METAL = 1,
    MAT_EMISSIVE = 2,
    MAT_WIREFRAME = 3,
    MAT_DIELECTRIC = 4
} MaterialKind;

typedef struct {
    MaterialKind kind;
    vec3 albedo;
    float fuzz;
    vec3 emit;
    float refractive_index;
} Material;

typedef struct {
    vec3 center;
    float radius;
    unsigned int material_idx;
} Sphere;


typedef struct {
    unsigned int index_start;
    unsigned int index_count;
    mat4 model;
    AABB box;
    unsigned int material_idx;
} Mesh;

EXTERN_C void rng_init(size_t image_width, size_t image_height, int seed);
EXTERN_C void rng_deinit(void);

#endif // RAYCAST_H_
