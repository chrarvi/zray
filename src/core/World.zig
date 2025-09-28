const std = @import("std");
const core = @import("core.zig");
const rc = @import("../gpu/raycast.zig");

const World = @This();

spheres: std.ArrayList(rc.Sphere),
mesh_atlas: core.MeshAtlas,
materials: std.ArrayList(rc.Material),
blas: core.BLASBuilder,
tlas: core.TLASBuilder,

pub fn init(allocator: std.mem.Allocator) !World {
    return .{
        .spheres = std.ArrayList(rc.Sphere).init(allocator),
        .mesh_atlas = core.MeshAtlas.init(allocator),
        .materials = std.ArrayList(rc.Material).init(allocator),
        .blas = core.BLASBuilder.init(allocator),
        .tlas = core.TLASBuilder.init(allocator),
    };
}

pub fn deinit(self: *World) void {
    self.spheres.deinit();
    self.mesh_atlas.deinit();
    self.materials.deinit();
    self.blas.deinit();
    self.tlas.deinit();
}

pub fn register_material(self: *World, mat: rc.Material) !u32 {
    const idx = self.materials.items.len;
    try self.materials.append(mat);
    return @as(u32, @intCast(idx));
}
