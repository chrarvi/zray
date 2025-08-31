#ifndef MATH_CUH_
#define MATH_CUH_

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include <math.h>
#include <cmath>

#define PI 3.14159265358979323846f

inline __device__ float norm(vec3 v) {
    return norm3df(v.x, v.y, v.z);
}
inline  __device__ float rnorm(vec3 v) {
    return rnorm3df(v.x, v.y, v.z);
}

inline  __device__ vec3 operator-(vec3 a, vec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

inline  __device__ vec3 operator-(vec3 a, float b) {
    return {a.x - b, a.y - b, a.z - b};
}

inline  __device__ vec3 operator+(vec3 a, vec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

inline  __device__ vec3 operator+(vec3 a, float b) {
    return {a.x + b, a.y + b, a.z + b};
}

inline  __device__ vec3 operator+(float a, vec3 b) {
    return {a + b.x, a + b.y, a + b.z};
}

inline  __device__ vec3 operator*(vec3 v, float c) {
    return {v.x * c, v.y * c, v.z * c};
}
inline  __device__ vec3 operator*(vec3 a, vec3 b) {
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}
inline  __device__ vec3 operator*(float a, vec3 b) {
    return {a * b.x, a * b.y, a * b.z};
}

inline  __device__ vec3 operator/(vec3 v, float c) {
    const float r = 1/c;
    return v * r;
}

inline  __device__ vec3 normalize(vec3 v) {
    return v / norm(v);
}

inline  __device__ float dot(vec3 a, vec3 b) {
    return a.x*b.x + a.y* b.y + a.z*b.z;
}

inline __device__ vec3 cross(vec3 a, vec3 b) {
    return vec3 {
        a.y*b.z - a.z*b.y,
        a.z*b.x - a.x*b.z,
        a.x*b.y - a.y*b.x
    };
}

inline  __device__ float clamp(float v, float mn, float mx) {
    return fmaxf(fminf(v, mx), mn);
}

inline  __device__ bool near_zero(vec3 v) {
    float s = 1e-8f;
    return (fabsf(v.x) < s) && (fabsf(v.y) < s) && (fabsf(v.z) < s);
}

inline  __device__ vec3 random_float3_uniform(curandState* local_state, float mn, float mx) {
    float c = mn + (mx - mn);
    float x = c * curand_uniform(local_state);
    float y = c * curand_uniform(local_state);
    float z = c * curand_uniform(local_state);

    return {x, y, z};
}

inline  __device__ vec3 random_unit_vector(curandState* local_state) {
    // pretty expensive but it's better than rejection sampling
    float u = curand_uniform(local_state);
    float theta = 2.0f * PI * curand_uniform(local_state);

    float z = 1.0f - 2.0f * u;
    float r = sqrtf(1.0f - z * z);

    float x = r * cosf(theta);
    float y = r * sinf(theta);

    return {x, y, z};
}

inline  __device__ vec3 random_on_hemisphere(curandState* local_state, const vec3 normal) {
    vec3 on_unit_sphere = random_unit_vector(local_state);
    if (dot(on_unit_sphere, normal) > 0.0) {
        return on_unit_sphere;
    } else {
        return -1.0 * on_unit_sphere;
    }
}

inline  __device__ vec3 reflect(vec3 v, vec3 n) {
    return v - 2.0f * dot(v, n) * n;
}

inline __device__ float4 mmul(const float M[4][4], const float4& v) {
    float4 r;
    r.x = M[0][0] * v.x + M[0][1] * v.y + M[0][2] * v.z + M[0][3] * v.w;
    r.y = M[1][0] * v.x + M[1][1] * v.y + M[1][2] * v.z + M[1][3] * v.w;
    r.z = M[2][0] * v.x + M[2][1] * v.y + M[2][2] * v.z + M[2][3] * v.w;
    r.w = M[3][0] * v.x + M[3][1] * v.y + M[3][2] * v.z + M[3][3] * v.w;
    return r;
}

#endif // MATH_CUH_
