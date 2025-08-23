#include <cmath>
#include <cuda_runtime.h>
#include <ios>
#include <math.h>

#include <curand.h>
#include <curand_kernel.h>

#define RNG_SEED 1234

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
    unsigned int samples_per_pixel;
    int max_depth;
} CameraData;

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
__device__ float3 operator*(float3 a, float3 b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
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

__device__ float3 random_float3_uniform(curandState* local_state, float mn, float mx) {
    float c = mn + (mx - mn);
    float x = c * curand_uniform(local_state);
    float y = c * curand_uniform(local_state);
    float z = c * curand_uniform(local_state);

    return make_float3(x, y, z);
}


#define PI 3.14159265358979323846f
__device__ float3 random_unit_vector(curandState* local_state) {
    // pretty expensive but it's better than rejection sampling
    float u = curand_uniform(local_state);
    float theta = 2.0f * PI * curand_uniform(local_state);

    float z = 1.0f - 2.0f * u;
    float r = sqrtf(1.0f - z * z);

    float x = r * cosf(theta);
    float y = r * sinf(theta);

    return make_float3(x, y, z);
}

__device__ float3 random_on_hemisphere(curandState* local_state, const float3 normal) {
    float3 on_unit_sphere = random_unit_vector(local_state);
    if (dot(on_unit_sphere, normal) > 0.0) {
        return on_unit_sphere;
    } else {
        return -1.0 * on_unit_sphere;
    }
}

__global__ void setup_rng(curandState* state, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    curand_init(RNG_SEED, idx, 0, &state[idx]);
}

__device__ bool range_constains(float v, float min, float max) {
    return min <= v && v <= max;
}

__device__ bool range_surrounds(float v, float min, float max) {
    return min < v && v < max;
}

__device__ float3 ray_at(const Ray* ray, float t) {
    return ray->origin + t * ray->dir;
}

__device__ float color_linear_to_gamma(float comp) {
    if (comp > 0.0f) return sqrtf(comp);
    return 0.0f;
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

__device__ float3 sample_square(curandState *local_state) {
    float x = curand_uniform(local_state) - 0.5f;
    float y = curand_uniform(local_state) - 0.5f;
    float z = 0.0f;
    return make_float3(x, y, z);
}

__device__ Ray sample_ray(curandState *local_state, unsigned int img_x, unsigned int img_y, float3 pixel00_loc, float3 pixel_delta_u, float3 pixel_delta_v, float3 ray_origin) {
    const float3 offset = sample_square(local_state);
    const float3 pixel_sample = pixel00_loc + ((img_x + offset.x) * pixel_delta_u) + ((img_y + offset.y) * pixel_delta_v);

    const float3 ray_direction = pixel_sample - ray_origin;
    return Ray{.origin = ray_origin, .dir = ray_direction};
}

__device__ float3 ray_color(const Ray& ray, int max_depth, const Sphere* spheres, unsigned int spheres_count, curandState* local_state) {
    Ray current_ray = ray;
    float3 attenuation = make_float3(1.0f, 1.0f, 1.0f);
    float3 color = make_float3(0.0f, 0.0f, 0.0f);

    for (int depth = 0; depth < max_depth; ++depth) {
        HitRecord hit_record;
        if (spheres_hit(&current_ray, spheres, spheres_count, 0.001f, INFINITY, &hit_record)) {
            // lambertian
            float3 dir = hit_record.normal + random_unit_vector(local_state);
            current_ray.origin = hit_record.point + 1e-4f * hit_record.normal;
            current_ray.dir = dir;
            attenuation = attenuation * 0.5f;
        } else {
            const float3 unit_dir = normalize(current_ray.dir);
            float t = 0.5f * (unit_dir.y + 1.0f);
            color = color + ((1.0f - t) * make_float3(1.0f, 1.0f, 1.0f) + t * make_float3(0.5f, 0.7f, 1.0f)) * attenuation;
            break;
        }
    }
    return color;
}

__global__ void render_kernel(unsigned char* img, const CameraData* cam, const Sphere* spheres, unsigned int spheres_count, curandState* rng_state) {
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

    curandState* local_state = &rng_state[y * cam->image_width + x];
    float3 color = make_float3(0.0f, 0.0f, 0.0f);
    for (size_t sample = 0u; sample < cam->samples_per_pixel; ++sample) {
        const Ray ray = sample_ray(local_state, x, y, pixel00_loc, pixel_delta_u, pixel_delta_v, camera_center);
        color = color + ray_color(ray, cam->max_depth, spheres, spheres_count, local_state);
    }

    color = color / (float)cam->samples_per_pixel;
    // gamma correction
    float r = color_linear_to_gamma(color.x);
    float g = color_linear_to_gamma(color.y);
    float b = color_linear_to_gamma(color.z);

    int idx = 3 * (y * cam->image_width + x);
    img[idx+0] = (unsigned char)(255.0f * clamp(r, 0.0f, 0.999f));
    img[idx+1] = (unsigned char)(255.0f * clamp(g, 0.0f, 0.999f));
    img[idx+2] = (unsigned char)(255.0f * clamp(b, 0.0f, 0.999f));
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

    render_kernel<<<grid, block>>>(d_img, cam, spheres, spheres_count, d_rng_state);

    cudaMemcpy(img, d_img, img_size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(d_img);
    cudaFree(d_rng_state);
}
