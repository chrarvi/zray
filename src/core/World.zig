const std = @import("std");
const core = @import("core.zig");
const rc = @import("../gpu/raycast.zig");

const World = @This();

spheres: std.ArrayList(rc.Sphere),
vb: core.HostVertexBuffer,

pub fn init(allocator: std.mem.Allocator) !World {
    return .{
        .spheres = std.ArrayList(rc.Sphere).init(allocator),
        .vb = core.HostVertexBuffer.init(allocator),
    };
}

pub fn deinit(self: *World) void {
    self.spheres.deinit();
    self.vb.deinit();
}
