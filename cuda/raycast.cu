#include <cmath>
#include <cuda_runtime.h>
#include <math.h>

#include <curand.h>
#include <curand_kernel.h>

#define RNG_SEED 1234

__device__ float norm(float3 v) {
    return norm3df(v.x, v.y, v.z);
}
__device__ float rnorm(float3 v) {
    return rnorm3df(v.x, v.y, v.z);
}

__device__ float3 operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 operator-(float3 a, float b) {
    return make_float3(a.x - b, a.y - b, a.z - b);
}

__device__ float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 operator+(float3 a, float b) {
    return make_float3(a.x + b, a.y + b, a.z + b);
}

__device__ float3 operator+(float a, float3 b) {
    return make_float3(a + b.x, a + b.y, a + b.z);
}

__device__ float3 operator*(float3 v, float c) {
    return make_float3(v.x * c, v.y * c, v.z * c);
}

__device__ float3 operator/(float3 v, float c) {
    const float r = 1/c;
    return v * r;
}

__device__ float3 normalize(float3 v) {
    return v / norm(v);
}

__device__ float3 operator*(float a, float3 b) {
    return make_float3(a * b.x, a * b.y, a * b.z);
}

__device__ float dot(float3 a, float3 b) {
    return a.x*b.x + a.y* b.y + a.z*b.z;
}

__device__ float clamp(float v, float mn, float mx) {
    return fmaxf(fminf(v, mx), mn);
}

__global__ void setup_rng(curandState* state, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    curand_init(RNG_SEED, idx, 0, &state[idx]);
}

typedef struct {
    float3 center;
    float radius;
} Sphere;

typedef struct {
    float3 origin;
    float3 dir;
} Ray;

typedef struct {
    float3 point;
    float3 normal;
    float t;
    bool front_face;
} HitRecord;

// Mirrored in zig
typedef struct {
    unsigned int image_width;
    unsigned int image_height;
    float focal_length;
} CameraData;

__device__ bool range_constains(float v, float min, float max) {
    return min <= v && v <= max;
}

__device__ bool range_surrounds(float v, float min, float max) {
    return min < v && v < max;
}

__device__ float3 ray_at(const Ray* ray, float t) {
    return ray->origin + t * ray->dir;
}

__device__ bool sphere_hit(const Sphere *sphere, const Ray *ray, float ray_tmin,
                           float ray_tmax, HitRecord *hit_record) {
    const float3 oc = sphere->center - ray->origin;
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

    const float3 point = ray_at(ray, root);
    hit_record->t = root;
    hit_record->point = point;

    const float3 outward_normal = (point - sphere->center) / sphere->radius;
    hit_record->front_face = dot(ray->dir, outward_normal) < 0.0;
    hit_record->normal = hit_record->front_face ? outward_normal : outward_normal * -1.0f;

    return true;
}

__device__ bool spheres_hit(const Ray* ray, const Sphere *spheres, unsigned int spheres_count, float ray_tmin, float ray_tmax, HitRecord* hit_record) {
    bool hit = false;
    float closest_so_far = ray_tmax;
    for (size_t i = 0u; i < spheres_count; ++i) {
        HitRecord temp_hit = {};
        const Sphere* sphere = &spheres[i];
        bool _hit = sphere_hit(sphere, ray, ray_tmin, closest_so_far, &temp_hit);
        if (_hit) {
            hit = true;
            closest_so_far = temp_hit.t;
            hit_record->t = temp_hit.t;
            hit_record->normal = temp_hit.normal;
            hit_record->point = temp_hit.point;
            hit_record->front_face = temp_hit.front_face;
        }
    }

    return hit;
}

__device__ float3 ray_color(const Ray* ray, const Sphere* spheres, unsigned int spheres_count) {
    HitRecord hit_record = {0};
    bool hit = spheres_hit(ray, spheres, spheres_count, 0.0f, INFINITY, &hit_record);
    if (hit) {
        return 0.5 * (hit_record.normal + 1.0f);
    }

    const float3 unit_dir = normalize(ray->dir);
    const float a = 0.5 * (unit_dir.y + 1.0);
    return (1.0f - a) + (a * make_float3(0.5, 0.7, 1.0));
}

__global__ void render_kernel(unsigned char* img, const CameraData* cam, const Sphere* spheres, unsigned int spheres_count) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cam->image_width || y >= cam->image_height) return;

    const float viewport_height = 2.0f;
    const float viewport_width = viewport_height * (float)cam->image_width / (float)cam->image_height;
    const float3 viewport_u = make_float3( viewport_width, 0.0, 0.0 );
    const float3 viewport_v = make_float3( 0.0, -viewport_height, 0.0);
    const float3 camera_center = make_float3(0.0f, 0.0f, 0.0f);
    const float3 pixel_delta_u = viewport_u / (float)cam->image_width;
    const float3 pixel_delta_v = viewport_v / (float)cam->image_height;
    const float3 viewport_upper_left = camera_center - make_float3(0.0f, 0.0f, cam->focal_length) - viewport_u / 2.0f - viewport_v / 2.0f;
    const float3 pixel00_loc = viewport_upper_left + (pixel_delta_u + pixel_delta_v) / 2.0f;
    const float3 pixel_center = pixel00_loc + (x * pixel_delta_u) + (y * pixel_delta_v);
    const float3 ray_direction = pixel_center - camera_center;

    const Ray ray = {.origin = camera_center, .dir = ray_direction};
    const float3 color = ray_color(&ray, spheres, spheres_count);
    int idx = 3 * (y * cam->image_width + x);
    img[idx+0] = (unsigned char)(255.0f * fminf(fmaxf(color.x, 0.0f), 1.0f));
    img[idx+1] = (unsigned char)(255.0f * fminf(fmaxf(color.y, 0.0f), 1.0f));
    img[idx+2] = (unsigned char)(255.0f * fminf(fmaxf(color.z, 0.0f), 1.0f));
}

__global__ void noise_kernel(unsigned char* img, const CameraData* cam, curandState* rng_state) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cam->image_width || y >= cam->image_height) return;

    int idx = y * cam->image_width + x;
    curandState local_state = rng_state[idx];

    float r = curand_uniform(&local_state);
    float g = curand_uniform(&local_state);
    float b = curand_uniform(&local_state);

    rng_state[idx] = local_state;

    int pixel_idx = 3 * idx;
    img[pixel_idx + 0] = (unsigned char)(255.0f * r);
    img[pixel_idx + 1] = (unsigned char)(255.0f * g);
    img[pixel_idx + 2] = (unsigned char)(255.0f * b);
}



extern "C" void launch_raycast(unsigned char *img, const CameraData* cam) {
    unsigned char *d_img;
    size_t img_size = cam->image_height * cam->image_width * 3U * sizeof(unsigned char);
    cudaMalloc((void**)&d_img, img_size);
    cudaMemcpy(d_img, img, img_size, cudaMemcpyHostToDevice);

    curandState *d_rng_state;
    cudaMalloc(&d_rng_state, cam->image_height * cam->image_width * sizeof(curandState));

    dim3 block(16, 16);
    dim3 grid((cam->image_width + block.x - 1) / block.x,
                (cam->image_height + block.y - 1) / block.y);

    setup_rng<<<grid, block>>>(d_rng_state, cam->image_width, cam->image_height);

    Sphere spheres[] = {
        {.center = {0.0f, 0.0f, -1.0f}, .radius = 0.5f},
        {.center = {0.0f, -100.5f, -1.0f}, .radius = 100.0f}
    };
    size_t spheres_count = sizeof(spheres) / sizeof(spheres[0]);

    float3 cam_center = make_float3(0.0f, 0.0f, 0.0f);

    render_kernel<<<grid, block>>>(d_img, cam, spheres, spheres_count);
    noise_kernel<<<grid, block>>>(d_img, cam, d_rng_state);

    cudaMemcpy(img, d_img, img_size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}
