const std = @import("std");
const rc = @import("raycast.zig");
const cu = @import("cuda.zig");
const gpu = @import("gpu.zig");

const DeviceWorld = @This();

spheres: cu.CudaBuffer(rc.Sphere),
vb: gpu.DeviceVertexBuffer,
indices: cu.CudaBuffer(u32),
mesh_ranges: cu.CudaBuffer(rc.Mesh),

pub fn init(spheres_capacity: usize, vertex_capacity: usize, indices_capactity: usize, mesh_capacity: usize) !DeviceWorld {
    return .{
        .spheres = try cu.CudaBuffer(rc.Sphere).init(spheres_capacity),
        .vb = try gpu.DeviceVertexBuffer.init(vertex_capacity),
        // Technically not correct since the number of indices is most
        // often less than the number of vertices, since shared vertices
        // are a thing in meshes.
        // TODO: be more explicit about the size of this cuda buffer
        .indices = try cu.CudaBuffer(u32).init(indices_capactity),
        .mesh_ranges = try cu.CudaBuffer(rc.Mesh).init(mesh_capacity),
    };
}

pub fn deinit(self: *DeviceWorld) void {
    self.spheres_dev.deinit();
    self.vb_dev.deinit();
    self.indices.deinit();
    self.mesh_ranges.deinit();
}
