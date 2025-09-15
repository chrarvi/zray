const std = @import("std");
const rc = @import("raycast.zig");
const cu = @import("cuda.zig");
const gpu = @import("gpu.zig");

const DeviceWorld = @This();

spheres: cu.CudaBuffer(rc.Sphere),
vb: gpu.DeviceVertexBuffer,

pub fn init(spheres_capacity: usize, vertex_capacity: usize) !DeviceWorld {
    return .{
        .spheres = try cu.CudaBuffer(rc.Sphere).init(spheres_capacity),
        .vb = try gpu.DeviceVertexBuffer.init(vertex_capacity),
    };
}

pub fn deinit(self: *DeviceWorld) void {
    self.spheres_dev.deinit();
    self.vb_dev.deinit();
}
