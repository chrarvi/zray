#include "stdio.h"
#include "assert.h"
#include "raycast.h"
#include <cmath>
#include <cuda_runtime.h>
#include <ios>
#include <math.h>
#include "math.cuh"
#include "tensor_view.cuh"

#include <stdint.h>

#include <stdio.h>

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
    float u, v;
} HitRecord;

typedef struct {
    bool did_scatter;
    Ray scattered_ray;
    vec3 attenuation;
    vec3 emission;
} ScatterResult;


typedef struct {
    TensorView<float, 2> pos;
    TensorView<float, 2> norm;
    TensorView<float, 2> color;
} VertexBuffers;


// TODO: Consider exposing this in the header and just preparing is on the zig
// side
typedef struct {
    CameraData *cam;
    TensorView<Sphere, 1> spheres;
    VertexBuffers vb;
    TensorView<uint32_t, 1> indices;
    TensorView<Mesh, 1> meshes;
    TensorView<Material, 1> materials;
} Scene;

__device__ inline vec3 tv_get_vec3(TensorView<float, 2> tv, size_t i) {
    return vec3{ tv.at(i, 0), tv.at(i, 1), tv.at(i, 2) };
}

__global__ void setup_rng(curandState* state, int width, int height, int seed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    curand_init(seed, idx, 0, &state[idx]);
}

__device__ vec3 ray_at(const Ray* ray, float t) {
    return ray->origin + t * ray->dir;
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
    float* out_t, float* out_u, float* out_v,
    vec3* out_normal, bool* out_front_face)
{
    const float EPS = 1e-8f;

    vec3 e1 = p2 - p1;
    vec3 e2 = p3 - p1;

    vec3 h = cross(ray->dir, e2);
    float a = dot(e1, h);
    if (fabsf(a) < EPS) return false;  // parallel

    float f = 1.0f / a;
    vec3 s = ray->origin - p1;
    float u = f * dot(s, h);
    if (u < 0.0f || u > 1.0f) return false;

    vec3 q = cross(s, e1);
    float v = f * dot(ray->dir, q);
    if (v < 0.0f || u + v > 1.0f) return false;

    float t = f * dot(e2, q);
    if (!range_surrounds(t, ray_tmin, ray_tmax)) return false;

    vec3 outward_normal = normalize(cross(e1, e2));
    bool front_face = dot(ray->dir, outward_normal) < 0.0f;

    if (out_t) *out_t = t;
    if (out_u) *out_u = u;
    if (out_v) *out_v = v;
    if (out_front_face) *out_front_face = front_face;
    if (out_normal) *out_normal = front_face ? outward_normal : outward_normal * -1.0f;
    return true;
}

__device__ bool mesh_hit(
    const Ray* ray,
    const VertexBuffers* vb,
    TensorView<uint32_t, 1> indices,
    TensorView<Mesh, 1> meshes,
    TensorView<Material, 1> materials,
    float ray_tmin, float ray_tmax,
    HitRecord* hit_record)
{
    const size_t n = vb->pos.shape[0];
    if (n < 3) return false;

    bool hit_anything = false;
    float closest = ray_tmax;

    for (size_t m = 0; m < meshes.shape[0]; ++m) {
        Mesh* mesh = &meshes.at(m);
        // todo: skip mesh if ray does not intersect the mesh bounding box at all

        uint32_t start = mesh->index_start;
        uint32_t end   = start + mesh->index_count;

        Material* material = &materials.at(mesh->material_idx);

        for (size_t i = start; i + 2 < end; i+=3) {
            uint32_t i0 = indices.at(i+0);
            uint32_t i1 = indices.at(i+1);
            uint32_t i2 = indices.at(i+2);

            vec3 p0 = tv_get_vec3(vb->pos, i0);
            vec3 p1 = tv_get_vec3(vb->pos, i1);
            vec3 p2 = tv_get_vec3(vb->pos, i2);

            vec4 p0_h = vec4{p0.x, p0.y, p0.z, 1.0};
            vec4 p1_h = vec4{p1.x, p1.y, p1.z, 1.0};
            vec4 p2_h = vec4{p2.x, p2.y, p2.z, 1.0};

            vec4 p0_world_h = lmmul(mesh->model, p0_h);
            vec4 p1_world_h = lmmul(mesh->model, p1_h);
            vec4 p2_world_h = lmmul(mesh->model, p2_h);

            vec3 p0_world = vec3{p0_world_h.x, p0_world_h.y, p0_world_h.z};
            vec3 p1_world = vec3{p1_world_h.x, p1_world_h.y, p1_world_h.z};
            vec3 p2_world = vec3{p2_world_h.x, p2_world_h.y, p2_world_h.z};

            float t, u, v;
            vec3 tri_normal;
            bool front_face;

            if (triangle_hit(p0_world, p1_world, p2_world, ray, ray_tmin, closest, &t, &u, &v, &tri_normal, &front_face)) {
                hit_anything = true;
                closest = t;
                vec3 point = ray_at(ray, t);

                vec3 n0 = tri_normal;
                vec3 n1 = tri_normal;
                vec3 n2 = tri_normal;
                if (vb->norm.shape[0] >= max(i0, max(i1, i2))) {
                    n0 = normalize(tv_get_vec3(vb->norm, i + 0));
                    n1 = normalize(tv_get_vec3(vb->norm, i + 1));
                    n2 = normalize(tv_get_vec3(vb->norm, i + 2));
                }
                vec3 n_shade = normalize(bary_lerp(n0, n1, n2, u, v));
                if (!front_face) n_shade = n_shade * -1;

                vec3 c0{1,1,1}, c1{1,1,1}, c2{1,1,1};
                if (vb->color.shape[0] >= i + 3) {
                    c0 = tv_get_vec3(vb->color, i + 0);
                    c1 = tv_get_vec3(vb->color, i + 1);
                    c2 = tv_get_vec3(vb->color, i + 2);
                }
                vec3 albedo = bary_lerp(c0, c1, c2, u, v);

                hit_record->t          = t;
                hit_record->point      = point;
                hit_record->normal     = n_shade;
                hit_record->front_face = front_face;
                hit_record->u = u;
                hit_record->v = v;

                hit_record->material = *material;
            }
        }
    }

    return hit_anything;
}


__device__ bool spheres_hit(const Ray* ray, TensorView<Sphere, 1> d_spheres, TensorView<Material, 1> d_materials, float ray_tmin, float ray_tmax, HitRecord* hit_record) {
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
            hit_record->material = d_materials.at(sphere->material_idx);
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

__device__ ScatterResult scatter_material(
    const Material& mat,
    const Ray& in_ray,
    const HitRecord& hit,
    curandState* rng)
{
    ScatterResult result;
    result.did_scatter = false;
    result.attenuation = vec3{1,1,1};
    result.emission    = mat.emit;

    switch (mat.kind) {
        case MAT_LAMBERTIAN: {
            vec3 scatter_dir = hit.normal + random_unit_vector(rng);
            if (near_zero(scatter_dir)) scatter_dir = hit.normal;

            result.scattered_ray = Ray{
                hit.point + 1e-4f * hit.normal,
                scatter_dir
            };
            result.attenuation = mat.albedo;
            result.did_scatter = true;
            break;
        }

        case MAT_METAL: {
            vec3 reflected = reflect(in_ray.dir, hit.normal);
            reflected = normalize(reflected) + mat.fuzz * random_unit_vector(rng);

            result.scattered_ray = Ray{
                hit.point + 1e-4f * hit.normal,
                reflected
            };
            result.attenuation = mat.albedo;
            result.did_scatter = (dot(result.scattered_ray.dir, hit.normal) > 0.0f);
            break;
        }

        case MAT_EMISSIVE: {
            result.did_scatter = false;
            break;
        }

        case MAT_WIREFRAME: {
            float u = hit.u;
            float v = hit.v;
            float w = 1.0f - u - v;
            const float edge_thickness = 0.02f;

            bool is_edge = (u < edge_thickness ||
                            v < edge_thickness ||
                            w < edge_thickness);

            result.did_scatter = false;
            result.attenuation = is_edge ? vec3{0,0,0} : mat.albedo;
            break;
        }
        case MAT_DIELECTRIC: {
            // Attenuation is always one, glass surface absorbs nothing.
            result.attenuation = {1.0, 1.0, 1.0};
            result.did_scatter = true;
            float ri = hit.front_face ? (1.0f / mat.refractive_index) : mat.refractive_index;
            vec3 unit_dir = normalize(in_ray.dir);
            float cos_theta = fminf(dot(-1.0 * unit_dir, hit.normal), 1.0);
            float sin_theta = sqrtf(1.0 - cos_theta * cos_theta);

            bool cannot_refract = ri * sin_theta > 1.0;

            vec3 direction;
            if (cannot_refract || reflectance(cos_theta, ri) > curand_uniform(rng)) {
                direction = reflect(unit_dir, hit.normal);
            } else {
                direction = refract(unit_dir, hit.normal, ri);
            }

            result.scattered_ray = Ray{
                hit.point + 1e-4f * direction,
                direction
            };
            break;
        }
    }

    return result;
}


__device__ vec3 ray_color(
    const Ray& ray,
    int max_depth,
    const Scene *scene,
    curandState* local_state)
{
    Ray current_ray = ray;
    vec3 throughput = {1.0f, 1.0f, 1.0f};
    vec3 accum = {0.0f, 0.0f, 0.0f};

    for (int depth = 0; depth < max_depth; ++depth) {
        HitRecord best_hit;
        bool hit_anything = false;
        float tmax = INFINITY;

        HitRecord sphere_hitrec;
        if (spheres_hit(&current_ray, scene->spheres, scene->materials, 0.001f, tmax, &sphere_hitrec)) {
            best_hit = sphere_hitrec;
            tmax = sphere_hitrec.t;
            hit_anything = true;
        }
        HitRecord mesh_hitrec;
        if (mesh_hit(&current_ray, &scene->vb, scene->indices, scene->meshes, scene->materials, 0.001f, tmax, &mesh_hitrec)) {
            best_hit = mesh_hitrec;
            tmax = mesh_hitrec.t;
            hit_anything = true;
        }

        if (hit_anything) {
            ScatterResult sr = scatter_material(best_hit.material, current_ray, best_hit, local_state);

            accum = accum + throughput * sr.emission;
            if (!sr.did_scatter) {
                accum = accum + throughput *sr.attenuation;
                break;
            }

            throughput = throughput * sr.attenuation;
            current_ray = sr.scattered_ray;
        } else {
            // miss → sky
            vec3 unit_dir = normalize(current_ray.dir);
            float t = 0.5f * (unit_dir.y + 1.0f);
            vec3 sky = (1.0f - t) * vec3{1.0f, 1.0f, 1.0f} + t * vec3{0.5f, 0.7f, 1.0f};
            accum = accum + throughput * sky;
            break;
        }
    }
    return accum;
}

__global__ void render_kernel(TensorView<float, 3> d_img_accum,
                              TensorView<char, 3> d_img, const CameraData *cam,
                              Scene scene, curandState *rng_state,
                              unsigned int frame_idx,
                              bool temporal_averaging
                              )
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cam->image_width || y >= cam->image_height) return;

    curandState* local_state = &rng_state[y * cam->image_width + x];
    vec3 color = {0.0f, 0.0f, 0.0f};

    for (size_t sample = 0u; sample < cam->samples_per_pixel; ++sample) {
        vec3 offset = sample_square(local_state);
        float ndc_x = ((x + offset.x) / (float)cam->image_width)  * 2.0f - 1.0f;
        float ndc_y = ((y + offset.y) / (float)cam->image_height) * 2.0f - 1.0f;

        ndc_y = -ndc_y;

        vec4 clip = vec4{ndc_x, ndc_y, -1.0f, 1.0f};

        vec4 cam_h = lmmul(cam->inv_proj, clip);
        vec3 dir_cam = normalize(vec3{ cam_h.x, cam_h.y, cam_h.z } / cam_h.w);

        vec4 origin_cam = vec4{0, 0, 0, 1};
        vec4 dir_cam4   = vec4{dir_cam.x, dir_cam.y, dir_cam.z, 0};

        vec4 origin_world4 = lmmul(cam->camera_to_world, origin_cam);
        vec4 dir_world4    = lmmul(cam->camera_to_world, dir_cam4);

        Ray ray = Ray {
            .origin = vec3{ origin_world4.x, origin_world4.y, origin_world4.z },
            .dir    = normalize({ dir_world4.x, dir_world4.y, dir_world4.z }),
        };

        color = color + ray_color(ray, cam->max_depth, &scene, local_state);
    }


    color = color / (float)cam->samples_per_pixel;

    // --- Temporal accumulation ---
    vec3 prev = vec3{
        d_img_accum.at(y, x, 0),
        d_img_accum.at(y, x, 1),
        d_img_accum.at(y, x, 2)
    };
    vec3 new_avg;
    if (temporal_averaging) {
        new_avg = (prev * frame_idx + color) / (frame_idx + 1);
        d_img_accum.at(y, x, 0) = new_avg.x;
        d_img_accum.at(y, x, 1) = new_avg.y;
        d_img_accum.at(y, x, 2) = new_avg.z;
    } else {
        new_avg = color;
    }

    // gamma corrected output
    vec3 display_color = clamp(color_linear_to_gamma(new_avg), 0.0, 0.999);
    d_img.at(y, x, 0) = (unsigned char)(255.0f * display_color.x);
    d_img.at(y, x, 1) = (unsigned char)(255.0f * display_color.y);
    d_img.at(y, x, 2) = (unsigned char)(255.0f * display_color.z);
}

curandState *d_rng_state;


EXTERN_C void rng_init(size_t image_height, size_t image_width, int seed) {
    CHECK_CUDA(cudaMalloc(&d_rng_state, image_height * image_width * sizeof(curandState)));

    dim3 block(32, 8);
    dim3 grid((image_width + block.x - 1) / block.x,
                (image_height + block.y - 1) / block.y);

    setup_rng<<<grid, block>>>(d_rng_state, image_width, image_height, seed);
    CHECK_CUDA(cudaPeekAtLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

EXTERN_C void launch_raycast(TensorView<float, 3> d_img_accum,
                             TensorView<char, 3> d_img, const CameraData *cam,
                             TensorView<Sphere, 1> d_spheres,
                             TensorView<float, 2> d_vb_pos,
                             TensorView<float, 2> d_vb_color,
                             TensorView<float, 2> d_vb_norm,
                             TensorView<uint32_t, 1> d_indices,
                             TensorView<Mesh, 1> d_meshes,
                             TensorView<Material, 1> d_materials,
                             unsigned int frame_idx,
                             bool temporal_averaging) {
    // d_img: height, width 3
    // d_spheres: n_spheres
    // d_vb_pos: n_vertex, 3
    // d_vb_color: n_vertex, 3
    // d_vb_norm: n_vertex, 3
    // d_indices: n_mesh_indices
    // d_meshes: n_meshes
    // d_materials: n_materials
    Scene scene = Scene{
        .spheres = d_spheres,
        .vb = VertexBuffers {
            .pos = d_vb_pos,
            .norm = d_vb_norm,
            .color = d_vb_color,
        },
        .indices = d_indices,
        .meshes = d_meshes,
        .materials = d_materials,
    };

    dim3 block(32, 8);
    dim3 grid((cam->image_width + block.x - 1) / block.x,
                (cam->image_height + block.y - 1) / block.y);

    render_kernel<<<grid, block>>>(d_img_accum, d_img, cam, scene, d_rng_state, frame_idx, temporal_averaging);
    CHECK_CUDA(cudaPeekAtLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

__global__ void clear_buffer(TensorView<float, 3> d_buf) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int height = d_buf.shape[0];
    unsigned int width = d_buf.shape[1];
    if (x >= width || y >= height) return;

    d_buf.at(y, x, 0) = 0.0;
    d_buf.at(y, x, 1) = 0.0;
    d_buf.at(y, x, 2) = 0.0;
}

EXTERN_C void launch_clear_buffer(TensorView<float, 3> d_buf) {
    dim3 block(32, 8);
    dim3 grid((d_buf.shape[1] + block.x - 1) / block.x,
                (d_buf.shape[0] + block.y - 1) / block.y);

    clear_buffer<<<grid, block>>>(d_buf);
    CHECK_CUDA(cudaPeekAtLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

EXTERN_C void rng_deinit(void) {
    CHECK_CUDA(cudaFree(d_rng_state));
}
