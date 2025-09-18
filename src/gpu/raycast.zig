const rc = @cImport(@cInclude("raycast.h"));
const cu = @import("cuda.zig");

pub const Sphere = rc.Sphere;
pub const CameraData = rc.CameraData;
pub const Material = rc.Material;
pub const Mesh = rc.Mesh;

pub const MaterialKind = struct {
    pub const Lambertian = rc.MAT_LAMBERTIAN;
    pub const Metal = rc.MAT_METAL;
    pub const Emissive = rc.MAT_EMISSIVE;
    pub const Wireframe = rc.MAT_WIREFRAME;
    pub const Dialectric = rc.MAT_DIELECTRIC;
};

pub const rng_init = rc.rng_init;
pub const rng_deinit = rc.rng_deinit;

pub extern fn launch_raycast(
    d_img: cu.TensorView(u8, 3),
    cam: *rc.CameraData,
    d_spheres: cu.TensorView(rc.Sphere, 1),
    d_vb_pos: cu.TensorView(f32, 2),
    d_vb_color: cu.TensorView(f32, 2),
    d_vb_normal: cu.TensorView(f32, 2),
    d_indices: cu.TensorView(u32, 1),
    d_meshes: cu.TensorView(rc.Mesh, 1),
    d_materials: cu.TensorView(rc.Material, 1),
) void;
