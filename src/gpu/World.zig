const std = @import("std");
const rc = @import("raycast.zig");
const cu = @import("cuda.zig");
const gpu = @import("gpu.zig");

const bvh = @import("../core/bvh.zig");
const HostBLAS = bvh.BLASBuilder;
const HostTLAS = bvh.TLASBuilder;

const DeviceWorld = @This();

spheres: cu.CudaBuffer(rc.Sphere),
vb: gpu.DeviceVertexBuffer,
indices: cu.CudaBuffer(u32),
mesh_ids: cu.CudaBuffer(u32),
meshes: cu.CudaBuffer(rc.Mesh),
partial_aabb: cu.CudaBuffer(rc.AABB),
materials: cu.CudaBuffer(rc.Material),
blas_meshinfo: cu.CudaBuffer(rc.BLASMeshInfo),
blas_nodes: cu.CudaBuffer(rc.BVHNode),
blas_prim_indices: cu.CudaBuffer(u32),
tlas_nodes: cu.CudaBuffer(rc.BVHNode),
tlas_prim_indices: cu.CudaBuffer(u32),

pub fn init(spheres_capacity: usize, vertex_capacity: usize, indices_capactity: usize, mesh_capacity: usize, materials_capacity: usize, blas_node_count: usize, tlas_node_count: usize) !DeviceWorld {
    return .{
        .spheres = try cu.CudaBuffer(rc.Sphere).init(spheres_capacity),
        .vb = try gpu.DeviceVertexBuffer.init(vertex_capacity),
        // Technically not correct since the number of indices is most
        // often less than the number of vertices, since shared vertices
        // are a thing in meshes.
        // TODO: be more explicit about the size of this cuda buffer
        .indices = try cu.CudaBuffer(u32).init(indices_capactity),
        .mesh_ids = try cu.CudaBuffer(u32).init(indices_capactity),
        .meshes = try cu.CudaBuffer(rc.Mesh).init(mesh_capacity),
        .materials = try cu.CudaBuffer(rc.Material).init(materials_capacity),

        .partial_aabb = try cu.CudaBuffer(rc.AABB).init(mesh_capacity*128),
        .blas_meshinfo = try cu.CudaBuffer(rc.BLASMeshInfo).init(mesh_capacity),
        .blas_nodes = try cu.CudaBuffer(rc.BVHNode).init(blas_node_count),
        .blas_prim_indices = try cu.CudaBuffer(u32).init(vertex_capacity / 3),
        .tlas_nodes = try cu.CudaBuffer(rc.BVHNode).init(tlas_node_count),
        .tlas_prim_indices = try cu.CudaBuffer(u32).init(mesh_capacity),
    };
}

pub fn deinit(self: *DeviceWorld) void {
    self.spheres.deinit();
    self.vb.deinit();
    self.indices.deinit();
    self.mesh_ids.deinit();
    self.meshes.deinit();
    self.materials.deinit();
    self.partial_aabb.deinit();
    self.blas_meshinfo.deinit();
    self.blas_nodes.deinit();
    self.blas_prim_indices.deinit();
    self.tlas_nodes.deinit();
    self.tlas_prim_indices.deinit();
}


// Convert host BVH nodes (which store left_idx and prims_offset separately)
// into device nodes (which pack them into a union keyed on prims_count) and
// upload them.
fn upload_nodes(dst: *cu.CudaBuffer(rc.BVHNode), src: []const bvh.BVHNode, alloc: std.mem.Allocator) !void {
    var temp_buffer = try std.array_list.Managed(rc.BVHNode).initCapacity(alloc, src.len);
    defer temp_buffer.deinit();

    for (src) |node| {
        var n = try temp_buffer.addOne();
        n.box = rc.AABB{
            .min = .{ .x = node.box.min.x, .y = node.box.min.y, .z = node.box.min.z },
            .max = .{ .x = node.box.max.x, .y = node.box.max.y, .z = node.box.max.z },
        };

        // Union: an internal node (prims_count == 0) stores its left child
        // index; a leaf stores its primitive offset.
        if (node.prims_count == 0) {
            n.lp.left_idx = @as(c_int, @intCast(node.left_idx));
        } else {
            n.lp.prims_offset = @as(c_uint, @intCast(node.prims_offset));
        }

        n.prims_count = @as(c_uint, @intCast(node.prims_count));
    }

    try dst.fromHost(temp_buffer.items);
}

pub fn bvh_to_device(self: *DeviceWorld, blas: *const HostBLAS, tlas: *const HostTLAS, alloc: std.mem.Allocator) !void {
    // BLAS: per-mesh node ranges + global triangle indices + per-mesh info.
    try upload_nodes(&self.blas_nodes, blas.nodes.items, alloc);
    try self.blas_prim_indices.fromHost(blas.prim_indices.items);
    try self.blas_meshinfo.fromHost(blas.meshinfo.items);

    // TLAS: instance BVH nodes + mesh (instance) indices.
    try upload_nodes(&self.tlas_nodes, tlas.nodes.items, alloc);
    try self.tlas_prim_indices.fromHost(tlas.prim_indices.items);
}
