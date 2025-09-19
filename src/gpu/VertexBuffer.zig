const std = @import("std");
const HostVertexBuffer = @import("../core/VertexBuffer.zig");
const cu = @import("cuda.zig");

const Self = @This();
pos_buf: cu.CudaBuffer(f32),
color_buf: cu.CudaBuffer(f32),
normal_buf: cu.CudaBuffer(f32),

pub fn init(vertex_capacity: usize) !Self {
    return .{
        .pos_buf= try cu.CudaBuffer(f32).init(vertex_capacity * 4),
        .color_buf= try cu.CudaBuffer(f32).init(vertex_capacity * 4),
        .normal_buf= try cu.CudaBuffer(f32).init(vertex_capacity * 4),
    };
}

pub fn deinit(self: *Self) void {
    self.pos_buf.deinit();
    self.color_buf.deinit();
    self.normal_buf.deinit();
}

pub fn fromHost(self: *Self, host_vb: *const HostVertexBuffer) !void {
    const pos_f32 = std.mem.bytesAsSlice(f32, std.mem.sliceAsBytes(host_vb.pos_buf.items));
    const col_f32 = std.mem.bytesAsSlice(f32, std.mem.sliceAsBytes(host_vb.color_buf.items));
    const norm_f32 = std.mem.bytesAsSlice(f32, std.mem.sliceAsBytes(host_vb.normal_buf.items));

    try self.pos_buf.fromHost(pos_f32);
    try self.color_buf.fromHost(col_f32);
    try self.normal_buf.fromHost(norm_f32);
}
