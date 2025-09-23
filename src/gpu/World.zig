const std = @import("std");
const rc = @import("raycast.zig");
const cu = @import("cuda.zig");
const gpu = @import("gpu.zig");

const HostBVH = @import("../core/bvh.zig").BVHBuilder;

const DeviceWorld = @This();

spheres: cu.CudaBuffer(rc.Sphere),
vb: gpu.DeviceVertexBuffer,
indices: cu.CudaBuffer(u32),
meshes: cu.CudaBuffer(rc.Mesh),
partial_aabb: cu.CudaBuffer(rc.AABB),
materials: cu.CudaBuffer(rc.Material),
bvh_nodes: cu.CudaBuffer(rc.BVHNode),
bvh_prim_indices: cu.CudaBuffer(u32),

pub fn init(spheres_capacity: usize, vertex_capacity: usize, indices_capactity: usize, mesh_capacity: usize, materials_capacity: usize, bvh_max_depth: usize) !DeviceWorld {
    const bvh_size = std.math.pow(usize, 2, bvh_max_depth + 1) - 1;
    return .{
        .spheres = try cu.CudaBuffer(rc.Sphere).init(spheres_capacity),
        .vb = try gpu.DeviceVertexBuffer.init(vertex_capacity),
        // Technically not correct since the number of indices is most
        // often less than the number of vertices, since shared vertices
        // are a thing in meshes.
        // TODO: be more explicit about the size of this cuda buffer
        .indices = try cu.CudaBuffer(u32).init(indices_capactity),
        .meshes = try cu.CudaBuffer(rc.Mesh).init(mesh_capacity),
        .materials = try cu.CudaBuffer(rc.Material).init(materials_capacity),

        .partial_aabb = try cu.CudaBuffer(rc.AABB).init(mesh_capacity*128),
        .bvh_nodes = try cu.CudaBuffer(rc.BVHNode).init(bvh_size),
        .bvh_prim_indices = try cu.CudaBuffer(u32).init(vertex_capacity / 3),
    };
}

pub fn deinit(self: *DeviceWorld) void {
    self.spheres.deinit();
    self.vb.deinit();
    self.indices.deinit();
    self.meshes.deinit();
    self.materials.deinit();
    self.partial_aabb.deinit();
    self.bvh_nodes.deinit();
    self.bvh_prim_indices.deinit();
}

pub fn bvh_to_device(self: *DeviceWorld, host: *const HostBVH, alloc: std.mem.Allocator) !void {
    var temp_buffer = try std.ArrayList(rc.BVHNode).initCapacity(alloc, host.nodes.items.len);
    defer temp_buffer.deinit();

    for (host.nodes.items) |node| {
        var n = try temp_buffer.addOne();
        n.box = rc.AABB {
            .min = .{ .x=node.box.min.x, .y=node.box.min.y, .z=node.box.min.z },
            .max = .{ .x=node.box.max.x, .y=node.box.max.y, .z=node.box.max.z },
        };

        n.left_idx = @as(c_int, @intCast(node.left_idx));
        n.right_idx = @as(c_int, @intCast(node.right_idx));
        n.prims_count = @as(c_uint, @intCast(node.prims_count));
        n.prims_offset = @as(c_uint, @intCast(node.prims_offset));
    }

    try self.bvh_nodes.fromHost(temp_buffer.items);
    try self.bvh_prim_indices.fromHost(host.prim_indices.items);
}
