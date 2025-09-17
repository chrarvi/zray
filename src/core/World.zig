const std = @import("std");
const core = @import("core.zig");
const rc = @import("../gpu/raycast.zig");

const World = @This();

spheres: std.ArrayList(rc.Sphere),
mesh_atlas: core.MeshAtlas,
materials: std.ArrayList(rc.Material),

pub fn init(allocator: std.mem.Allocator) !World {
    return .{
        .spheres = std.ArrayList(rc.Sphere).init(allocator),
        .mesh_atlas = core.MeshAtlas.init(allocator),
        .materials = std.ArrayList(rc.Material).init(allocator),
    };
}

pub fn deinit(self: *World) void {
    self.spheres.deinit();
    self.mesh_atlas.deinit();
    self.materials.deinit();
}
