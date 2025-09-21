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

typedef struct {
    AABB box;   // 2*3*4 = 24 bytes
    int left_idx;  // 4 bytes
    int right_idx;  // 4 bytes
    unsigned int prims_offset;  // 4 bytes
    unsigned int prims_count;  // 4 bytes
} BVHNode;  // total 40 bytes
// a cache line is 128 bytes, so we are not aligned here
// if we could hit 32 bytes we'd be golden. We can do that by removing
// right idx since it's always left_idx + 1, and then doing the
// following trick:
//
// Because an internal node always has children but no triangles, and a leaf
// node never has children and always triangles, we can store them in the same
// field
//
// typedef struct {
//     AABB box;
//     union {
//         int left_idx;
//         unsigned int prims_offset;
//     };
//     unsigned int prims_count;
// } BVHNode;
//
// When using the struct we can then check prims_count to determine if
// it's an internal node (and thus access left_idx) or a leaf node
// (and thus access prims_offset)
//
// TODO: implement the above to get good cache alignment!

EXTERN_C void rng_init(size_t image_width, size_t image_height, int seed);
EXTERN_C void rng_deinit(void);

#endif // RAYCAST_H_
