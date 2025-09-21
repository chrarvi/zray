const rc = @cImport(@cInclude("raycast.h"));
const cu = @import("cuda.zig");

pub const Sphere = rc.Sphere;
pub const CameraData = rc.CameraData;
pub const Material = rc.Material;
pub const Mesh = rc.Mesh;
pub const AABB = rc.AABB;
pub const BVHNode = rc.BVHNode;

pub const MaterialKind = struct {
    pub const Lambertian = rc.MAT_LAMBERTIAN;
    pub const Metal = rc.MAT_METAL;
    pub const Emissive = rc.MAT_EMISSIVE;
    pub const Wireframe = rc.MAT_WIREFRAME;
    pub const Dielectric = rc.MAT_DIELECTRIC;
};

pub const rng_init = rc.rng_init;
pub const rng_deinit = rc.rng_deinit;

pub extern fn launch_raycast(
    d_img_accum: cu.TensorView(f32, 3),
    d_img: cu.TensorView(u8, 3),
    cam: *rc.CameraData,
    d_spheres: cu.TensorView(Sphere, 1),
    d_vb_pos: cu.TensorView(f32, 2),
    d_vb_color: cu.TensorView(f32, 2),
    d_vb_normal: cu.TensorView(f32, 2),
    d_indices: cu.TensorView(u32, 1),
    d_meshes: cu.TensorView(Mesh, 1),
    d_materials: cu.TensorView(Material, 1),
    d_bvh_nodes: cu.TensorView(BVHNode, 1),
    d_bvh_prim_indices: cu.TensorView(u32, 1),
    frame_idx: u32,
    temporal_averaging: bool,
) void;

pub extern fn model_to_world(
    d_vb_pos: cu.TensorView(f32, 2), // (num_vertices, 3)
    d_vb_normal: cu.TensorView(f32, 2),
    d_indices: cu.TensorView(u32, 1),
    d_meshes: cu.TensorView(rc.Mesh, 1),
) void;

pub extern fn compute_aabb(
    d_vb_pos: cu.TensorView(f32, 2),
    d_indices: cu.TensorView(u32, 1),
    d_meshes: cu.TensorView(rc.Mesh, 1),
    d_partial_aabb: cu.TensorView(rc.AABB, 1),
) void;

pub extern fn launch_clear_buffer(
    d_img_accum: cu.TensorView(f32, 3),
) void;
