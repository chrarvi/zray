const std = @import("std");

// FFI
const cudaSuccess: c_int = 0;

pub fn checkCuda(ret: c_int) !void {
    if (ret == 0) {
        return;
    } else {
        std.debug.print("Unknown cuda error caught {}", .{ret});
        return error.cudaUnknownError;
    }
}

pub const cudaMemcpyHostToDevice: c_int = 1;
pub const cudaMemcpyDeviceToHost: c_int = 2;

pub extern fn cudaMalloc(ptr: *?*anyopaque, size: usize) c_int;
pub extern fn cudaFree(ptr: *anyopaque) c_int;
pub extern fn cudaMemcpy(dst: *anyopaque, src: *const anyopaque, size: usize, kind: c_int) c_int;
pub extern fn cudaDeviceSynchronize() c_int;

pub fn CudaBuffer(comptime ValueT: type) type {
    const BufferT = [*c]ValueT;
    return struct {
        const Self = @This();
        dev_ptr: BufferT = null,
        len: usize = 0,

        pub fn init(len: usize) !Self {
            var buf = Self{};
            var raw: ?*anyopaque = null;
            const size_bytes = @sizeOf(ValueT) * len;

            try checkCuda(cudaMalloc(&raw, size_bytes));
            buf.dev_ptr = @as([*c]ValueT, @ptrCast(@alignCast(raw.?)));
            buf.len = len;
            return buf;
        }

        pub fn deinit(self: *Self) !void {
            try checkCuda(cudaFree(@as(*anyopaque, @ptrCast(self.dev_ptr))));
            self.dev_ptr = null;
            self.len = 0;
        }

        pub fn fromHost(self: *Self, src: []const ValueT) !void {
            if (src.len > self.len) {
                return error.BufferOverflow;
            }
            try checkCuda(cudaMemcpy(
                self.dev_ptr,
                src.ptr,
                src.len * @sizeOf(ValueT),
                cudaMemcpyHostToDevice,
            ));
        }
        pub fn toHost(self: *const Self, dst: []ValueT) !void {
            if (dst.len > self.len) {
                return error.BufferOverflow;
            }
            try checkCuda(cudaMemcpy(
                @as(*anyopaque, @ptrCast(dst.ptr)),
                @as(*anyopaque, @ptrCast(self.dev_ptr)),
                self.len * @sizeOf(ValueT),
                cudaMemcpyDeviceToHost,
            ));
        }
    };
}
