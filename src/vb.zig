const std = @import("std");
const al = @import("math.zig");
const cu = @import("cuda.zig");

pub const HostVertexBuffer = struct {
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
};

pub const DeviceVertexBuffer = struct {
    const Self = @This();
    pos_buf: cu.CudaBuffer(f32),
    color_buf: cu.CudaBuffer(f32),
    normal_buf: cu.CudaBuffer(f32),

    pub fn init(capacity: usize) !Self {
        return .{
            .pos_buf= try cu.CudaBuffer(f32).init(capacity),
            .color_buf= try cu.CudaBuffer(f32).init(capacity),
            .normal_buf= try cu.CudaBuffer(f32).init(capacity),
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
};
