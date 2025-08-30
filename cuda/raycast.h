// in lib.h

#ifndef RAYCAST_H_
#define RAYCAST_H_

#include <stdlib.h>

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
    unsigned int image_width;
    unsigned int image_height;
    float focal_length;
    unsigned int samples_per_pixel;
    int max_depth;

    mat4 camera_to_world;
} CameraData;

typedef enum {
    MAT_LAMBERTIAN = 0,
    MAT_METAL = 1,
    MAT_EMISSIVE = 2
} MaterialKind;

typedef struct {
    MaterialKind kind;
    vec3 albedo;
    float fuzz;
    vec3 emit;
} Material;

typedef struct {
    vec3 center;
    float radius;
    Material material;
} Sphere;

typedef struct {
    vec3* p_buf;
    vec3* n_buf;
    vec3* c_buf;

    size_t count;
} VertexBuffer;

// VB
EXTERN_C VertexBuffer *vb_alloc(size_t count);
EXTERN_C void vb_free(VertexBuffer* vb);

EXTERN_C void init_cuda(const CameraData *cam, size_t spheres_count, int seed);
EXTERN_C void update_spheres(const Sphere *spheres, size_t spheres_count);
EXTERN_C void launch_raycast(unsigned char *img, const CameraData* cam, const Sphere* spheres, size_t spheres_count);
EXTERN_C void cleanup_cuda(void);

#endif // RAYCAST_H_
