const std = @import("std");
const al = @import("linalg.zig");
const cu = @import("../gpu/cuda.zig");

const Self = @This();
pos_buf: std.ArrayList(al.Vec3),
color_buf: std.ArrayList(al.Vec3),
normal_buf: std.ArrayList(al.Vec3),

pub fn init(allocator: std.mem.Allocator) Self {
    return .{
        .pos_buf= std.ArrayList(al.Vec3).init(allocator),
        .color_buf= std.ArrayList(al.Vec3).init(allocator),
        .normal_buf= std.ArrayList(al.Vec3).init(allocator),
    };
}

pub fn deinit(self: *Self) void {
    self.pos_buf.deinit();
    self.color_buf.deinit();
    self.normal_buf.deinit();
}

pub fn push_vertex(self: *Self, pos: al.Vec3, color: al.Vec3, normal: al.Vec3) !void {
    try self.pos_buf.append(pos);
    try self.color_buf.append(color);
    try self.normal_buf.append(normal);
}
