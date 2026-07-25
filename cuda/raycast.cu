#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>

#include "assert.h"
#include "math.cuh"
#include "raycast.h"
#include "stdio.h"
#include "tensor_view.cuh"

#define CHECK_CUDA(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr,                                              \
                    "CUDA Error: %s (error code %d) in %s at line %d\n", \
                    cudaGetErrorString(err), err, __FILE__, __LINE__);   \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
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
    uint32_t i0, i1, i2;
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
    CameraData* cam;
    TensorView<Sphere, 1> spheres;
    VertexBuffers vb;
    TensorView<uint32_t, 1> indices;
    TensorView<uint32_t, 1> mesh_ids;
    TensorView<Mesh, 1> meshes;
    TensorView<Material, 1> materials;
    TensorView<BLASMeshInfo, 1> blas_meshinfo;
    TensorView<BVHNode, 1> blas_nodes;
    TensorView<uint32_t, 1> blas_prim_indices;
    TensorView<BVHNode, 1> tlas_nodes;
    TensorView<uint32_t, 1> tlas_prim_indices;
} Scene;

__device__ inline vec3 tv_get_vec3(TensorView<float, 2> tv, size_t i) {
    return vec3{tv.at(i, 0), tv.at(i, 1), tv.at(i, 2)};
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

__device__ bool sphere_hit(const Sphere* sphere, const Ray* ray, float ray_tmin,
                           float ray_tmax, HitRecord* hit_record) {
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
    hit_record->normal =
        hit_record->front_face ? outward_normal : outward_normal * -1.0f;

    return true;
}

__device__ bool ray_triangle_hit(const vec3 p1, const vec3 p2, const vec3 p3,
                                 const Ray* ray, float ray_tmin, float ray_tmax,
                                 HitRecord* out) {
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

    out->t = t;
    out->u = u;
    out->v = v;
    out->front_face = front_face;
    out->normal = front_face ? outward_normal : outward_normal * -1.0f;
    out->point = ray_at(ray, t);
    return true;
}

// SLAB aabb intersection test. On hit, writes the ray parameter at which the
// ray enters the box (clamped to >= 0) to *t_near, so the BVH traversal can
// order children front-to-back and prune boxes that lie beyond the closest hit.
__device__ bool ray_aabb_hit(const Ray* ray, const AABB* box, float* t_near) {
    vec3 recip_dir = 1.0f / ray->dir;

    vec3 t_low = (box->min - ray->origin) * recip_dir;
    vec3 t_high = (box->max - ray->origin) * recip_dir;

    vec3 t_vec_close = fminf(t_low, t_high);
    vec3 t_vec_far = fmaxf(t_low, t_high);

    float t_close = fmaxf(t_vec_close);
    float t_far = fminf(t_vec_far);

    *t_near = fmaxf(t_close, 0.0f);
    return (t_close <= t_far) && (t_far >= 0.0f);
}

// Debug: total number of ray/AABB and ray/triangle intersection tests performed
// during a launch. Threads accumulate locally and do a single atomicAdd.
__device__ unsigned long long g_test_count;

// Intersect a single mesh instance's BLAS in that mesh's model space.
// `ray_model` is the world ray transformed by mesh->inv_model (direction left
// UN-normalized so that the ray parameter t is identical in world and model
// space). Updates *t_closest and, on a closer hit, fills *out with the hit in
// model space (t, u, v, front_face, model-space normal, i0/i1/i2).
__device__ bool blas_hit(const BLASMeshInfo* info, TensorView<BVHNode, 1> blas_nodes,
                         TensorView<uint32_t, 1> blas_prim_indices,
                         VertexBuffers const* vb, TensorView<uint32_t, 1> indices,
                         Ray const* ray_model, float ray_tmin, float* t_closest,
                         HitRecord* out, unsigned long long* nt) {
    bool hit_anything = false;

    // Stack entries carry the box entry distance so we can discard a node whose
    // whole box lies beyond the closest hit found so far (t_closest may shrink
    // after the node was pushed).
    const int MAX_STACK = 64;
    struct Entry { int idx; float t_near; };
    Entry stack[MAX_STACK];
    int stack_count = 0;
    stack[stack_count++] = {(int)info->node_offset, ray_tmin};  // this mesh's BLAS root

    while (stack_count > 0) {
        Entry entry = stack[--stack_count];
        if (entry.t_near > *t_closest) continue;  // pruned by a closer hit
        const BVHNode* node = &blas_nodes.at(entry.idx);

        if (node->prims_count > 0) {
            // Leaf: intersect triangles (model space).
            for (int i = node->lp.prims_offset;
                 i < node->lp.prims_offset + node->prims_count; ++i) {
                uint32_t tri_index = blas_prim_indices.at(i);  // global triangle index
                uint32_t i0 = indices.at(tri_index * 3 + 0);
                uint32_t i1 = indices.at(tri_index * 3 + 1);
                uint32_t i2 = indices.at(tri_index * 3 + 2);

                vec3 p0 = tv_get_vec3(vb->pos, i0);
                vec3 p1 = tv_get_vec3(vb->pos, i1);
                vec3 p2 = tv_get_vec3(vb->pos, i2);

                HitRecord tris_hit;
                (*nt)++;
                if (ray_triangle_hit(p0, p1, p2, ray_model, ray_tmin, *t_closest,
                                     &tris_hit)) {
                    hit_anything = true;
                    *t_closest = tris_hit.t;

                    vec3 n0 = tv_get_vec3(vb->norm, i0);
                    vec3 n1 = tv_get_vec3(vb->norm, i1);
                    vec3 n2 = tv_get_vec3(vb->norm, i2);
                    // Model-space smooth normal; the caller transforms it to
                    // world space.
                    out->normal = bary_lerp(n0, n1, n2, tris_hit.u, tris_hit.v);

                    out->t = tris_hit.t;
                    out->front_face = tris_hit.front_face;
                    out->u = tris_hit.u;
                    out->v = tris_hit.v;
                    out->i0 = i0;
                    out->i1 = i1;
                    out->i2 = i2;
                }
            }
        } else {
            int left = node->lp.left_idx;
            int right = node->lp.left_idx + 1;
            float t_left, t_right;
            (*nt) += 2;
            bool hit_left = ray_aabb_hit(ray_model, &blas_nodes.at(left).box, &t_left) &&
                            t_left <= *t_closest;
            bool hit_right = ray_aabb_hit(ray_model, &blas_nodes.at(right).box, &t_right) &&
                             t_right <= *t_closest;

            // Push the farther child first so the nearer one is popped (and gets
            // to shrink t_closest) first.
            if (hit_left && hit_right) {
                if (t_left <= t_right) {
                    stack[stack_count++] = {right, t_right};
                    stack[stack_count++] = {left, t_left};
                } else {
                    stack[stack_count++] = {left, t_left};
                    stack[stack_count++] = {right, t_right};
                }
            } else if (hit_left) {
                stack[stack_count++] = {left, t_left};
            } else if (hit_right) {
                stack[stack_count++] = {right, t_right};
            }
        }
    }

    return hit_anything;
}

// Two-level (TLAS over instances -> per-instance BLAS) traversal. The incoming
// ray is in world space. On the closest hit, *out is filled entirely in world
// space (point, normal).
__device__ bool ray_bvh_hit(TensorView<BLASMeshInfo, 1> blas_meshinfo,
                            TensorView<BVHNode, 1> blas_nodes,
                            TensorView<uint32_t, 1> blas_prim_indices,
                            TensorView<BVHNode, 1> tlas_nodes,
                            TensorView<uint32_t, 1> tlas_prim_indices,
                            TensorView<uint32_t, 1> mesh_ids,
                            TensorView<Mesh, 1> meshes, Ray const* ray,
                            VertexBuffers const* vb,
                            TensorView<uint32_t, 1> indices, float ray_tmin,
                            float ray_tmax, HitRecord* out, unsigned long long* nt) {
    float t_closest = ray_tmax;
    bool hit_anything = false;

    const int MAX_STACK = 64;
    struct Entry { int idx; float t_near; };
    Entry stack[MAX_STACK];
    int stack_count = 0;
    stack[stack_count++] = {0, ray_tmin};  // TLAS root

    while (stack_count > 0) {
        Entry entry = stack[--stack_count];
        if (entry.t_near > t_closest) continue;  // pruned by a closer hit
        const BVHNode* node = &tlas_nodes.at(entry.idx);

        if (node->prims_count > 0) {
            // Leaf: mesh instances.
            for (int p = node->lp.prims_offset;
                 p < node->lp.prims_offset + node->prims_count; ++p) {
                uint32_t mesh_id = tlas_prim_indices.at(p);
                const Mesh* mesh = &meshes.at(mesh_id);
                const BLASMeshInfo* info = &blas_meshinfo.at(mesh_id);

                // Transform the world ray into this instance's model space.
                // The direction is intentionally NOT normalized so that the ray
                // parameter t matches between world and model space.
                vec4 ray_o_h = {ray->origin.x, ray->origin.y, ray->origin.z, 1.0f};
                vec4 ray_d_h = {ray->dir.x, ray->dir.y, ray->dir.z, 0.0f};
                vec4 ray_o_model = mat4_lmmul(mesh->inv_model, ray_o_h);
                vec4 ray_d_model = mat4_lmmul(mesh->inv_model, ray_d_h);
                Ray ray_model = {
                    .origin = {ray_o_model.x, ray_o_model.y, ray_o_model.z},
                    .dir = {ray_d_model.x, ray_d_model.y, ray_d_model.z},
                };

                HitRecord blas_rec;
                if (blas_hit(info, blas_nodes, blas_prim_indices, vb, indices,
                             &ray_model, ray_tmin, &t_closest, &blas_rec, nt)) {
                    hit_anything = true;

                    // Bring the hit back to world space. Because t is preserved,
                    // the world-space point is just the world ray evaluated at t.
                    // The normal transforms by the inverse-transpose of the
                    // model matrix, i.e. transpose(inv_model) * n, which is
                    // mat3_rmmul(n, inv_model).
                    vec3 n_world =
                        normalize(mat3_rmmul(blas_rec.normal, mesh->inv_model));
                    if (!blas_rec.front_face) n_world = n_world * -1.0f;

                    out->t = blas_rec.t;
                    out->point = ray_at(ray, blas_rec.t);
                    out->normal = n_world;
                    out->front_face = blas_rec.front_face;
                    out->u = blas_rec.u;
                    out->v = blas_rec.v;
                    out->i0 = blas_rec.i0;
                    out->i1 = blas_rec.i1;
                    out->i2 = blas_rec.i2;
                }
            }
        } else {
            int left = node->lp.left_idx;
            int right = node->lp.left_idx + 1;
            float t_left, t_right;
            (*nt) += 2;
            bool hit_left = ray_aabb_hit(ray, &tlas_nodes.at(left).box, &t_left) &&
                            t_left <= t_closest;
            bool hit_right = ray_aabb_hit(ray, &tlas_nodes.at(right).box, &t_right) &&
                             t_right <= t_closest;
            if (hit_left && hit_right) {
                if (t_left <= t_right) {
                    stack[stack_count++] = {right, t_right};
                    stack[stack_count++] = {left, t_left};
                } else {
                    stack[stack_count++] = {left, t_left};
                    stack[stack_count++] = {right, t_right};
                }
            } else if (hit_left) {
                stack[stack_count++] = {left, t_left};
            } else if (hit_right) {
                stack[stack_count++] = {right, t_right};
            }
        }
    }

    return hit_anything;
}

__device__ bool mesh_hit(const Ray* ray, const VertexBuffers* vb,
                         TensorView<uint32_t, 1> indices,
                         TensorView<Mesh, 1> meshes,
                         TensorView<Material, 1> materials, float ray_tmin,
                         float ray_tmax, HitRecord* hit_record) {
    const size_t n = vb->pos.shape[0];
    if (n < 3) return false;

    bool hit_anything = false;
    float t_closest = ray_tmax;

    for (size_t m = 0; m < meshes.shape[0]; ++m) {
        Mesh* mesh = &meshes.at(m);
        // if (!ray_aabb_hit(ray, &mesh->box)) {
        //     continue;
        // }

        uint32_t start = mesh->index_start;
        uint32_t end = start + mesh->index_count;

        Material* material = &materials.at(mesh->material_idx);

        for (size_t i = start; i + 2 < end; i += 3) {
            uint32_t i0 = indices.at(i + 0);
            uint32_t i1 = indices.at(i + 1);
            uint32_t i2 = indices.at(i + 2);

            vec3 p0_world = tv_get_vec3(vb->pos, i0);
            vec3 p1_world = tv_get_vec3(vb->pos, i1);
            vec3 p2_world = tv_get_vec3(vb->pos, i2);

            HitRecord tris_hit = {};
            if (ray_triangle_hit(p0_world, p1_world, p2_world, ray, ray_tmin,
                                 t_closest, &tris_hit)) {
                hit_anything = true;
                t_closest = tris_hit.t;

                vec3 n0 = normalize(tv_get_vec3(vb->norm, i0));
                vec3 n1 = normalize(tv_get_vec3(vb->norm, i1));
                vec3 n2 = normalize(tv_get_vec3(vb->norm, i2));
                vec3 n_shade =
                    normalize(bary_lerp(n0, n1, n2, tris_hit.u, tris_hit.v));

                if (!tris_hit.front_face) n_shade = n_shade * -1;

                hit_record->t = tris_hit.t;
                hit_record->point = tris_hit.point;
                hit_record->normal = n_shade;
                hit_record->front_face = tris_hit.front_face;
                hit_record->u = tris_hit.u;
                hit_record->v = tris_hit.v;
                hit_record->material = *material;
                hit_record->i0 = i0;
                hit_record->i1 = i1;
                hit_record->i2 = i2;
            }
        }
    }

    return hit_anything;
}

__device__ bool spheres_hit(const Ray* ray, TensorView<Sphere, 1> d_spheres,
                            TensorView<Material, 1> d_materials, float ray_tmin,
                            float ray_tmax, HitRecord* hit_record) {
    bool hit = false;
    float closest_so_far = ray_tmax;
    for (size_t i = 0u; i < d_spheres.shape[0]; ++i) {
        HitRecord temp_hit = {};
        const Sphere* sphere = &d_spheres.at(i);
        bool _hit =
            sphere_hit(sphere, ray, ray_tmin, closest_so_far, &temp_hit);
        if (_hit) {
            hit = true;
            closest_so_far = temp_hit.t;
            hit_record->t = temp_hit.t;
            hit_record->material = d_materials.at(sphere->material_idx);
            hit_record->normal = temp_hit.normal;
            hit_record->point = temp_hit.point;
            hit_record->front_face = temp_hit.front_face;
            hit_record->i0 = i;
            hit_record->i1 = i;
            hit_record->i2 = i;
        }
    }

    return hit;
}

__device__ vec3 sample_square(curandState* local_state) {
    float x = curand_uniform(local_state) - 0.5f;
    float y = curand_uniform(local_state) - 0.5f;
    float z = 0.0f;
    return {x, y, z};
}

__device__ ScatterResult scatter_material(const Material& mat,
                                          const Ray& in_ray,
                                          const HitRecord& hit,
                                          curandState* rng) {
    ScatterResult result;
    result.did_scatter = false;
    result.attenuation = vec3{1, 1, 1};
    result.emission = mat.emit;

    switch (mat.kind) {
        case MAT_LAMBERTIAN: {
            vec3 scatter_dir = hit.normal + random_unit_vector(rng);
            if (near_zero(scatter_dir)) scatter_dir = hit.normal;

            result.scattered_ray =
                Ray{hit.point + 1e-4f * hit.normal, scatter_dir};
            result.attenuation = mat.albedo;
            result.did_scatter = true;
            break;
        }

        case MAT_METAL: {
            vec3 reflected = reflect(in_ray.dir, hit.normal);
            reflected =
                normalize(reflected) + mat.fuzz * random_unit_vector(rng);

            result.scattered_ray =
                Ray{hit.point + 1e-4f * hit.normal, reflected};
            result.attenuation = mat.albedo;
            result.did_scatter =
                (dot(result.scattered_ray.dir, hit.normal) > 0.0f);
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

            bool is_edge = (u < edge_thickness || v < edge_thickness ||
                            w < edge_thickness);

            result.did_scatter = false;
            result.attenuation = is_edge ? vec3{0, 0, 0} : mat.albedo;
            break;
        }
        case MAT_DIELECTRIC: {
            // Attenuation is always one, glass surface absorbs nothing.
            result.attenuation = {1.0, 1.0, 1.0};
            result.did_scatter = true;
            float ri = hit.front_face ? (1.0f / mat.refractive_index)
                                      : mat.refractive_index;
            vec3 unit_dir = normalize(in_ray.dir);
            float cos_theta = fminf(dot(-1.0 * unit_dir, hit.normal), 1.0);
            float sin_theta = sqrtf(1.0 - cos_theta * cos_theta);

            bool cannot_refract = ri * sin_theta > 1.0;

            vec3 direction;
            if (cannot_refract ||
                reflectance(cos_theta, ri) > curand_uniform(rng)) {
                direction = reflect(unit_dir, hit.normal);
            } else {
                direction = refract(unit_dir, hit.normal, ri);
            }

            result.scattered_ray =
                Ray{hit.point + 1e-4f * direction, direction};
            break;
        }
    }

    return result;
}

__device__ vec3 ray_color(const Ray& ray, int max_depth, Scene* scene,
                          curandState* local_state, unsigned long long* nt) {
    Ray current_ray = ray;
    vec3 throughput = {1.0f, 1.0f, 1.0f};
    vec3 accum = {0.0f, 0.0f, 0.0f};

    for (int depth = 0; depth < max_depth; ++depth) {
        HitRecord best_hit;
        bool hit_anything = false;
        float tmax = INFINITY;

        HitRecord sphere_hitrec;
        if (spheres_hit(&current_ray, scene->spheres, scene->materials, 0.001f,
                        tmax, &sphere_hitrec)) {
            best_hit = sphere_hitrec;
            tmax = sphere_hitrec.t;
            hit_anything = true;
        }

        // HitRecord mesh_hitrec;
        // if (mesh_hit(&current_ray, &scene->vb, scene->indices, scene->meshes,
        //              scene->materials, 0.001f, tmax, &mesh_hitrec)) {
        //     best_hit = mesh_hitrec;
        //     tmax = mesh_hitrec.t;
        //     hit_anything = true;
        // }

        HitRecord bvh_hitrec;
        if (ray_bvh_hit(scene->blas_meshinfo, scene->blas_nodes,
                        scene->blas_prim_indices, scene->tlas_nodes,
                        scene->tlas_prim_indices, scene->mesh_ids,
                        scene->meshes, &current_ray, &scene->vb, scene->indices,
                        0.001f, tmax, &bvh_hitrec, nt)) {
            const uint32_t mesh_idx = scene->mesh_ids.at(bvh_hitrec.i0);
            const Mesh* mesh = &scene->meshes.at(mesh_idx);
            bvh_hitrec.material = scene->materials.at(mesh->material_idx);
            best_hit = bvh_hitrec;
            tmax = bvh_hitrec.t;
            hit_anything = true;
        }

        if (hit_anything) {
            ScatterResult sr = scatter_material(best_hit.material, current_ray,
                                                best_hit, local_state);

            accum = accum + throughput * sr.emission;
            if (!sr.did_scatter) {
                accum = accum + throughput * sr.attenuation;
                break;
            }

            throughput = throughput * sr.attenuation;
            current_ray = sr.scattered_ray;
        } else {
            vec3 unit_dir = normalize(current_ray.dir);
            float t = 0.5f * (unit_dir.y + 1.0f);
            vec3 sky = (1.0f - t) * vec3{1.0f, 1.0f, 1.0f} +
                       t * vec3{0.5f, 0.7f, 1.0f};
            accum = accum + throughput * sky;
            break;
        }
    }
    return accum;
}

__global__ void render_kernel(TensorView<float, 3> d_img_accum,
                              TensorView<char, 3> d_img, const CameraData* cam,
                              Scene scene, curandState* rng_state,
                              unsigned int frame_idx, bool temporal_averaging) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cam->image_width || y >= cam->image_height) return;

    curandState* local_state = &rng_state[y * cam->image_width + x];
    vec3 color = {0.0f, 0.0f, 0.0f};

    unsigned long long tests = 0;
    for (size_t sample = 0u; sample < cam->samples_per_pixel; ++sample) {
        vec3 offset = sample_square(local_state);
        float ndc_x = ((x + offset.x) / (float)cam->image_width) * 2.0f - 1.0f;
        float ndc_y = ((y + offset.y) / (float)cam->image_height) * 2.0f - 1.0f;

        ndc_y = -ndc_y;

        vec4 clip = vec4{ndc_x, ndc_y, -1.0f, 1.0f};

        vec4 cam_h = mat4_lmmul(cam->inv_proj, clip);
        vec3 dir_cam = normalize(vec3{cam_h.x, cam_h.y, cam_h.z} / cam_h.w);

        vec4 origin_cam = vec4{0, 0, 0, 1};
        vec4 dir_cam4 = vec4{dir_cam.x, dir_cam.y, dir_cam.z, 0};

        vec4 origin_world4 = mat4_lmmul(cam->camera_to_world, origin_cam);
        vec4 dir_world4 = mat4_lmmul(cam->camera_to_world, dir_cam4);

        Ray ray = Ray{
            .origin = vec3{origin_world4.x, origin_world4.y, origin_world4.z},
            .dir = normalize({dir_world4.x, dir_world4.y, dir_world4.z}),
        };

        color = color + ray_color(ray, cam->max_depth, &scene, local_state, &tests);
    }

    color = color / (float)cam->samples_per_pixel;
    atomicAdd(&g_test_count, tests);

    // --- Temporal accumulation ---
    vec3 prev = vec3{d_img_accum.at(y, x, 0), d_img_accum.at(y, x, 1),
                     d_img_accum.at(y, x, 2)};
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

curandState* d_rng_state;

EXTERN_C void rng_init(size_t image_height, size_t image_width, int seed) {
    CHECK_CUDA(cudaMalloc(&d_rng_state,
                          image_height * image_width * sizeof(curandState)));

    dim3 block(32, 8);
    dim3 grid((image_width + block.x - 1) / block.x,
              (image_height + block.y - 1) / block.y);

    setup_rng<<<grid, block>>>(d_rng_state, image_width, image_height, seed);
    CHECK_CUDA(cudaPeekAtLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

EXTERN_C void launch_raycast(
    TensorView<float, 3> d_img_accum, TensorView<char, 3> d_img,
    const CameraData* cam, TensorView<Sphere, 1> d_spheres,
    TensorView<float, 2> d_vb_pos, TensorView<float, 2> d_vb_color,
    TensorView<float, 2> d_vb_norm, TensorView<uint32_t, 1> d_indices,
    TensorView<uint32_t, 1> d_mesh_ids, TensorView<Mesh, 1> d_meshes,
    TensorView<Material, 1> d_materials,
    TensorView<BLASMeshInfo, 1> d_blas_meshinfo,
    TensorView<BVHNode, 1> d_blas_nodes,
    TensorView<uint32_t, 1> d_blas_prim_indices,
    TensorView<BVHNode, 1> d_tlas_nodes,
    TensorView<uint32_t, 1> d_tlas_prim_indices, unsigned int frame_idx,
    bool temporal_averaging, unsigned long long* out_test_count) {
    // d_img: height, width 3
    // d_spheres: n_spheres
    // d_vb_pos: n_vertex, 3
    // d_vb_color: n_vertex, 3
    // d_vb_norm: n_vertex, 3
    // d_indices: n_mesh_indices
    // d_meshes: n_meshes
    // d_materials: n_materials
    Scene scene = Scene{.spheres = d_spheres,
                        .vb =
                            VertexBuffers{
                                .pos = d_vb_pos,
                                .norm = d_vb_norm,
                                .color = d_vb_color,
                            },
                        .indices = d_indices,
                        .mesh_ids = d_mesh_ids,
                        .meshes = d_meshes,
                        .materials = d_materials,
                        .blas_meshinfo = d_blas_meshinfo,
                        .blas_nodes = d_blas_nodes,
                        .blas_prim_indices = d_blas_prim_indices,
                        .tlas_nodes = d_tlas_nodes,
                        .tlas_prim_indices = d_tlas_prim_indices};

    dim3 block(32, 8);
    dim3 grid((cam->image_width + block.x - 1) / block.x,
              (cam->image_height + block.y - 1) / block.y);

    unsigned long long zero = 0;
    CHECK_CUDA(cudaMemcpyToSymbol(g_test_count, &zero, sizeof(zero)));

    render_kernel<<<grid, block>>>(d_img_accum, d_img, cam, scene, d_rng_state,
                                   frame_idx, temporal_averaging);
    CHECK_CUDA(cudaPeekAtLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    if (out_test_count != nullptr) {
        CHECK_CUDA(cudaMemcpyFromSymbol(out_test_count, g_test_count,
                                        sizeof(*out_test_count)));
    }
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

EXTERN_C void rng_deinit(void) { CHECK_CUDA(cudaFree(d_rng_state)); }

__global__ void model_to_world_kernel(TensorView<float, 2> d_vb_pos,
                                      TensorView<float, 2> d_vb_norm,
                                      TensorView<uint32_t, 1> d_indices,
                                      TensorView<Mesh, 1> d_meshes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_vb_pos.shape[0]) return;

    // Find which mesh this index belongs to
    // (linear search works but is slow; better: precompute a per-index mesh_id
    // array)
    Mesh mesh;
    for (int m = 0; m < d_meshes.shape[0]; ++m) {
        uint32_t start = d_meshes.at(m).index_start;
        uint32_t end = start + d_meshes.at(m).index_count;
        if (idx >= start && idx < end) {
            mesh = d_meshes.at(m);
            break;
        }
    }

    // Transform position
    vec3 pos = tv_get_vec3(d_vb_pos, idx);
    vec4 pos_h = vec4{pos.x, pos.y, pos.z, 1.0f};
    vec4 pos_w = mat4_lmmul(mesh.model, pos_h);
    d_vb_pos.at(idx, 0) = pos_w.x;
    d_vb_pos.at(idx, 1) = pos_w.y;
    d_vb_pos.at(idx, 2) = pos_w.z;

    if (d_vb_norm.shape[0] > idx) {
        vec3 n = tv_get_vec3(d_vb_norm, idx);

        mat4 inv, normal_mat;
        if (mat4_inverse(mesh.model, inv)) {
            mat4_transpose(inv, normal_mat);

            // apply upper-left 3×3
            vec3 n_w = normalize(mat3_lmmul(normal_mat, n));
            d_vb_norm.at(idx, 0) = n_w.x;
            d_vb_norm.at(idx, 1) = n_w.y;
            d_vb_norm.at(idx, 2) = n_w.z;
        }
    }
}

EXTERN_C void model_to_world(TensorView<float, 2> d_vb_pos,
                             TensorView<float, 2> d_vb_norm,
                             TensorView<uint32_t, 1> d_indices,
                             TensorView<Mesh, 1> d_meshes) {
    dim3 block(256);
    dim3 grid((d_vb_pos.shape[0] + block.x - 1) / block.x);

    model_to_world_kernel<<<grid, block>>>(d_vb_pos, d_vb_norm, d_indices,
                                           d_meshes);

    CHECK_CUDA(cudaPeekAtLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

/// AABB stuff
__device__ AABB aabb_init() {
    return {.min = {FLT_MAX, FLT_MAX, FLT_MAX},
            .max = {-FLT_MAX, -FLT_MAX, -FLT_MAX}};
}

__device__ bool aabb_valid(const AABB& b) {
    return b.min.x <= b.max.x && b.min.y <= b.max.y && b.min.z <= b.max.z;
}

__device__ void aabb_extend(AABB& box, vec3 p) {
    box.min = fminf(box.min, p);
    box.max = fmaxf(box.max, p);
}

__device__ void aabb_merge(AABB& box, const AABB& other) {
    if (!aabb_valid(other)) return;
    aabb_extend(box, other.min);
    aabb_extend(box, other.max);
}

__global__ void compute_aabb_kernel_stage1(
    TensorView<float, 2> d_vb_pos, TensorView<uint32_t, 1> d_indices,
    TensorView<Mesh, 1> d_meshes, TensorView<AABB, 1> d_partial_boxes) {
    unsigned int block_id = blockIdx.x;
    unsigned int thread_id = threadIdx.x;
    unsigned int global_tid = block_id * blockDim.x + thread_id;
    unsigned int stride = blockDim.x * gridDim.x;

    extern __shared__ AABB sdata[];  // one per thread for local reductions

    for (unsigned int mesh_idx = 0; mesh_idx < d_meshes.shape[0]; ++mesh_idx) {
        const Mesh& mesh = d_meshes.at(mesh_idx);

        // start local AABB
        AABB local_box = aabb_init();

        for (unsigned int i = global_tid; i < mesh.index_count; i += stride) {
            uint32_t vi = d_indices.at(mesh.index_start + i);
            vec3 p = tv_get_vec3(d_vb_pos, vi);
            aabb_extend(local_box, p);
        }

        sdata[thread_id] = local_box;
        __syncthreads();

        // intra-block reduction
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (thread_id < s) {
                aabb_merge(sdata[thread_id], sdata[thread_id + s]);
            }
            __syncthreads();
        }

        // thread 0 writes final block AABB for this mesh
        if (thread_id == 0) {
            d_partial_boxes.at(mesh_idx * gridDim.x + block_id) = sdata[0];
        }
        __syncthreads();  // ensure clean slate before next mesh
    }
}

__global__ void compute_aabb_kernel_stage2(TensorView<AABB, 1> d_partial_boxes,
                                           TensorView<Mesh, 1> d_meshes,
                                           unsigned int num_blocks) {
    unsigned int mesh_idx = blockIdx.x;
    unsigned int tid = threadIdx.x;

    extern __shared__ AABB sdata[];

    // load one partial box per thread (if available)
    if (tid < num_blocks) {
        sdata[tid] = d_partial_boxes.at(mesh_idx * num_blocks + tid);
    } else {
        sdata[tid] = aabb_init();
    }
    __syncthreads();

    // reduce
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            aabb_merge(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_meshes.at(mesh_idx).box = sdata[0];
    }
}

/// Parallel reduction of mesh bounding boxes.
EXTERN_C void compute_aabb(TensorView<float, 2> d_vb_pos,
                           TensorView<uint32_t, 1> d_indices,
                           TensorView<Mesh, 1> d_meshes,
                           TensorView<AABB, 1> d_partial_aabb) {
    unsigned int num_vertices = d_vb_pos.shape[0];
    unsigned int threads = 256;
    unsigned int blocks = (num_vertices + threads - 1) / threads;

    // stage 1: one AABB per block per mesh
    size_t shmem_stage1 = threads * sizeof(AABB);
    compute_aabb_kernel_stage1<<<blocks, threads, shmem_stage1>>>(
        d_vb_pos, d_indices, d_meshes, d_partial_aabb);
    CHECK_CUDA(cudaPeekAtLastError());

    // stage 2: reduce block AABBs per mesh
    unsigned int threads_stage2 = 128;
    size_t shmem_stage2 = threads_stage2 * sizeof(AABB);
    compute_aabb_kernel_stage2<<<d_meshes.shape[0], threads_stage2,
                                 shmem_stage2>>>(d_partial_aabb, d_meshes,
                                                 blocks);
    CHECK_CUDA(cudaPeekAtLastError());

    CHECK_CUDA(cudaDeviceSynchronize());
}
