#include "stdio.h"
#include "assert.h"
#include "raycast.h"
#include <cmath>
#include <cuda_runtime.h>
#include <math.h>
#include "math.cuh"
#include "tensor_view.cuh"

#include <curand.h>
#include <curand_kernel.h>

#define CHECK_CUDA(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA Error: %s (error code %d) in %s at line %d\n", \
                    cudaGetErrorString(err), err, __FILE__, __LINE__);           \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

typedef struct {
    vec3 origin;
    vec3 dir;
} Ray;

typedef struct {
    vec3 point;
    vec3 normal;
    Material material;
    float t;
    bool front_face;
} HitRecord;

__global__ void setup_rng(curandState* state, int width, int height, int seed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    curand_init(seed, idx, 0, &state[idx]);
}

__device__ bool range_constains(float v, float min, float max) {
    return min <= v && v <= max;
}

__device__ bool range_surrounds(float v, float min, float max) {
    return min < v && v < max;
}

__device__ vec3 ray_at(const Ray* ray, float t) {
    return ray->origin + t * ray->dir;
}

__device__ float color_linear_to_gamma(float comp) {
    if (comp > 0.0f) return sqrtf(comp);
    return 0.0f;
}

__device__ bool scatter_lambertian(const Ray* ray, const HitRecord* hit_record, curandState* local_state, vec3* attenuation, Ray* scattered) {
    vec3 scatter_dir = hit_record->normal + random_unit_vector(local_state);
    if (near_zero(scatter_dir)) {
        scatter_dir = hit_record->normal;
    }

    scattered->dir = scatter_dir;
    attenuation->x = attenuation->x * hit_record->material.albedo.x;
    attenuation->y = attenuation->y * hit_record->material.albedo.y;
    attenuation->z = attenuation->z * hit_record->material.albedo.z;
    return true;
}

__device__ bool scatter_metal(const Ray* ray, const HitRecord* hit_record, curandState* local_state, vec3* attenuation, Ray* scattered) {
    vec3 reflected = reflect(ray->dir, hit_record->normal);
    reflected = normalize(reflected) + (hit_record->material.fuzz * random_unit_vector(local_state));
    scattered->dir = reflected;
    attenuation->x = attenuation->x * hit_record->material.albedo.x;
    attenuation->y = attenuation->y * hit_record->material.albedo.y;
    attenuation->z = attenuation->z * hit_record->material.albedo.z;
    return dot(scattered->dir, hit_record->normal) > 0.0f;
}

__device__ bool sphere_hit(const Sphere *sphere, const Ray *ray, float ray_tmin,
                           float ray_tmax, HitRecord *hit_record) {
    const vec3 oc = sphere->center - ray->origin;
    const float a = dot(ray->dir, ray->dir);
    const float h = dot(ray->dir, oc);
    const float c = dot(oc, oc) - sphere->radius * sphere->radius;
    const float disc = h * h - a * c;
    if (disc < 0.0f) {
        return false;
    }

    const float sqrtd = sqrtf(disc);
    float root = (h - sqrtd) / a;
    if (!range_surrounds(root, ray_tmin, ray_tmax)) {
        root = (h + sqrtd) / a;
        if (!range_surrounds(root, ray_tmin, ray_tmax)) {
            return false;
        }
    }

    const vec3 point = ray_at(ray, root);
    hit_record->t = root;
    hit_record->point = point;

    const vec3 outward_normal = (point - sphere->center) / sphere->radius;
    hit_record->front_face = dot(ray->dir, outward_normal) < 0.0;
    hit_record->normal = hit_record->front_face ? outward_normal : outward_normal * -1.0f;

    return true;
}

__device__ bool triangle_hit(
    const vec3 p1, const vec3 p2, const vec3 p3,
    const Ray *ray,
    float ray_tmin, float ray_tmax,
    HitRecord *hit_record)
{
    const float EPS = 1e-8f;

    vec3 e1 = p2 - p1;
    vec3 e2 = p3 - p1;

    vec3 h = cross(ray->dir, e2);
    float a = dot(e1, h);

    if (fabsf(a) < EPS) return false;  // Ray parallel to triangle

    float f = 1.0f / a;
    vec3 s = ray->origin - p1;
    float u = f * dot(s, h);
    if (u < 0.0f || u > 1.0f) return false;

    vec3 q = cross(s, e1);
    float v = f * dot(ray->dir, q);
    if (v < 0.0f || u + v > 1.0f) return false;

    float t = f * dot(e2, q);
    if (!range_surrounds(t, ray_tmin, ray_tmax)) return false;

    vec3 point = ray_at(ray, t);

    vec3 outward_normal = normalize(cross(e1, e2));
    bool front_face = dot(ray->dir, outward_normal) < 0.0f;

    hit_record->t = t;
    hit_record->point = point;
    hit_record->front_face = front_face;
    hit_record->normal = front_face ? outward_normal : outward_normal * -1.0f;

    return true;
}

__device__ bool spheres_hit(const Ray* ray, TensorView<Sphere, 1> d_spheres, float ray_tmin, float ray_tmax, HitRecord* hit_record) {
    bool hit = false;
    float closest_so_far = ray_tmax;
    for (size_t i = 0u; i < d_spheres.shape[0]; ++i) {
        HitRecord temp_hit = {};
        const Sphere* sphere = &d_spheres.at(i);
        bool _hit = sphere_hit(sphere, ray, ray_tmin, closest_so_far, &temp_hit);
        if (_hit) {
            hit = true;
            closest_so_far = temp_hit.t;
            hit_record->t = temp_hit.t;
            hit_record->material = sphere->material;
            hit_record->normal = temp_hit.normal;
            hit_record->point = temp_hit.point;
            hit_record->front_face = temp_hit.front_face;
        }
    }

    return hit;
}

__device__ bool triangles_hit(const Ray* ray, const VertexBuffer *d_vb, float ray_tmin, float ray_tmax, HitRecord* hit_record) {
    bool hit = false;
    float closest_so_far = ray_tmax;
    for (size_t i = 0u; i < d_vb->count; i+=3) {
        HitRecord temp_hit = {};
        const vec3 p0 = d_vb->p_buf[i];
        const vec3 p1 = d_vb->p_buf[i+1];
        const vec3 p2 = d_vb->p_buf[i+2];
        bool _hit = triangle_hit(p0, p1, p2, ray, ray_tmin, closest_so_far, &temp_hit);
        if (_hit) {
            hit = true;
            closest_so_far = temp_hit.t;
            hit_record->t = temp_hit.t;
            // hit_record->material = sphere->material;
            hit_record->normal = temp_hit.normal;
            hit_record->point = temp_hit.point;
            hit_record->front_face = temp_hit.front_face;
        }
    }

    return hit;
}

__device__ vec3 sample_square(curandState *local_state) {
    float x = curand_uniform(local_state) - 0.5f;
    float y = curand_uniform(local_state) - 0.5f;
    float z = 0.0f;
    return {x, y, z};
}

__device__ vec3 ray_color(const Ray& ray, int max_depth, TensorView<Sphere, 1> d_spheres, curandState* local_state) {
    Ray current_ray = ray;
    vec3 attenuation = {1.0f, 1.0f, 1.0f};
    vec3 color = {0.0f, 0.0f, 0.0f};

    for (int depth = 0; depth < max_depth; ++depth) {
        HitRecord hit_record;
        if (spheres_hit(&current_ray, d_spheres, 0.001f, INFINITY, &hit_record)) {
            bool scattered = false;
            Ray temp_ray = {current_ray.origin, current_ray.dir};
            switch (hit_record.material.kind) {
            case MAT_LAMBERTIAN:
                scattered = scatter_lambertian(&current_ray, &hit_record, local_state, &attenuation, &temp_ray);
                break;
            case MAT_METAL:
                scattered = scatter_metal(&current_ray, &hit_record, local_state, &attenuation, &temp_ray);
                break;
            case MAT_EMISSIVE:
                return color + attenuation * hit_record.material.emit;
            }
            if (scattered) {
                current_ray.origin = hit_record.point + 1e-4f * hit_record.normal;
                current_ray.dir = temp_ray.dir;
            } else {
                color = {0.0f, 0.0f, 0.0f};
                break;
            }
        } else {
            const vec3 unit_dir = normalize(current_ray.dir);
            float t = 0.5f * (unit_dir.y + 1.0f);
            color = color + ((1.0f - t) * vec3{1.0f, 1.0f, 1.0f} + t * vec3{0.5f, 0.7f, 1.0f}) * attenuation;
            break;
        }
    }
    return color;
}

__global__ void render_kernel(TensorView<char, 3> d_img, const CameraData* cam, TensorView<Sphere, 1> d_spheres, curandState* rng_state) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cam->image_width || y >= cam->image_height) return;

    const float viewport_height = 2.0f;
    const float viewport_width  = viewport_height * (float)cam->image_width / (float)cam->image_height;

    const vec3 pixel_delta_u = {viewport_width / (float)cam->image_width, 0.0f, 0.0f};
    const vec3 pixel_delta_v = {0.0f, -viewport_height / (float)cam->image_height, 0.0f};

    const vec3 viewport_upper_left = {-viewport_width / 2.0f, viewport_height / 2.0f, -cam->focal_length};
    const vec3 pixel00_loc = viewport_upper_left + (pixel_delta_u + pixel_delta_v) * 0.5f;

    curandState* local_state = &rng_state[y * cam->image_width + x];
    vec3 color = {0.0f, 0.0f, 0.0f};

    for (size_t sample = 0u; sample < cam->samples_per_pixel; ++sample) {
        vec3 offset = sample_square(local_state);
        vec3 pixel_sample = pixel00_loc + (x + offset.x) * pixel_delta_u + (y + offset.y) * pixel_delta_v;

        float4 origin_cam = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
        float4 dir_cam    = make_float4(pixel_sample.x, pixel_sample.y, pixel_sample.z, 0);

        float4 origin_world4 = mmul(cam->camera_to_world, origin_cam);
        float4 dir_world4    = mmul(cam->camera_to_world, dir_cam);

        Ray ray = Ray {
            .origin = vec3{origin_world4.x, origin_world4.y, origin_world4.z},
            .dir    = normalize({dir_world4.x, dir_world4.y, dir_world4.z}),
        };

        color = color + ray_color(ray, cam->max_depth, d_spheres, local_state);
    }

    color = color / (float)cam->samples_per_pixel;

    // gamma correction
    float r = color_linear_to_gamma(color.x);
    float g = color_linear_to_gamma(color.y);
    float b = color_linear_to_gamma(color.z);

    d_img.at(y, x, 0) = (unsigned char)(255.0f * clamp(r, 0.0f, 0.999f));
    d_img.at(y, x, 1) = (unsigned char)(255.0f * clamp(g, 0.0f, 0.999f));
    d_img.at(y, x, 2) = (unsigned char)(255.0f * clamp(b, 0.0f, 0.999f));
}

curandState *d_rng_state;


EXTERN_C void rng_init(size_t image_height, size_t image_width, int seed) {
    CHECK_CUDA(cudaMalloc(&d_rng_state, image_height * image_width * sizeof(curandState)));

    dim3 block(16, 16);
    dim3 grid((image_width + block.x - 1) / block.x,
                (image_height + block.y - 1) / block.y);

    setup_rng<<<grid, block>>>(d_rng_state, image_width, image_height, seed);
    CHECK_CUDA(cudaPeekAtLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

EXTERN_C void launch_raycast(TensorView<char, 3> d_img, const CameraData* cam, TensorView<Sphere, 1> d_spheres) {
    dim3 block(16, 16);
    dim3 grid((cam->image_width + block.x - 1) / block.x,
                (cam->image_height + block.y - 1) / block.y);

    render_kernel<<<grid, block>>>(d_img, cam, d_spheres, d_rng_state);
    CHECK_CUDA(cudaPeekAtLastError());
}

EXTERN_C void rng_deinit(void) {
    CHECK_CUDA(cudaFree(d_rng_state));
}
