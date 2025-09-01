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

/// Generic view struct for tensors. This struct is shared with CUDA,
/// who has a device side implementation for conveniently indexing these
/// tensors.
pub fn TensorView(comptime ValueT: type, comptime Rank: usize) type {
    return extern struct {
        const Self = @This();

        data: [*c]ValueT,
        shape: [Rank]usize,
        strides: [Rank]usize,

        pub fn init(buf: *const CudaBuffer(ValueT), shape: [Rank]usize) Self {
            const td = TensorDims(Rank).init(shape);
            return .{
                .data = buf.dev_ptr,
                .shape = td.dims,
                .strides = td.strides,
            };
        }

        pub fn at(self: *const Self, idx: [Rank]usize) *ValueT {
            var offset: usize = 0;
            inline for (0..Rank) |i| {
                offset += idx[i] * self.strides[i];
            }
            return self.data + offset;
        }
    };
}

pub fn TensorDims(comptime Rank: usize) type {
    return struct {
        dims: [Rank]usize,
        strides: [Rank]usize,

        pub fn init(dims: [Rank]usize) @This() {
            var strides: [Rank]usize = undefined;
            strides[Rank-1] = 1;
            var i = Rank - 1;
            while (i > 0) : (i -= 1) {
                strides[i-1] = strides[i] * dims[i];
            }
            return .{ .dims = dims, .strides = strides };
        }
    };
}

pub fn CudaBuffer(comptime ValueT: type) type {
    return struct {
        const Self = @This();
        const BufferT = [*c]ValueT;

        dev_ptr: BufferT = null,
        len: usize = 0,

        pub fn init(len: usize) !Self {
            var buf = Self{};
            buf.len = len;

            const size_bytes = @sizeOf(ValueT) * len;
            var raw: ?*anyopaque = null;
            try checkCuda(cudaMalloc(&raw, size_bytes));
            buf.dev_ptr = @as([*c]ValueT, @ptrCast(@alignCast(raw.?)));

            return buf;
        }

        pub fn deinit(self: *Self) void {
            checkCuda(cudaFree(@as(*anyopaque, @ptrCast(self.dev_ptr)))) catch |err| {
                std.debug.panic("Unable to free cuda memory, likely double free: {}", .{err});
            };
            self.dev_ptr = null;
            self.len = 0;
        }

        pub fn fromHost(self: *Self, src: []const ValueT) !void {
            if (src.len != self.len) return error.BufferSizeMismatch;
            try checkCuda(cudaMemcpy(
                self.dev_ptr,
                src.ptr,
                src.len * @sizeOf(ValueT),
                cudaMemcpyHostToDevice,
            ));
        }

        pub fn toHost(self: *const Self, dst: []ValueT) !void {
            if (dst.len != self.len) return error.BufferSizeMismatch;
            try checkCuda(cudaMemcpy(
                @as(*anyopaque, @ptrCast(dst.ptr)),
                @as(*anyopaque, @ptrCast(self.dev_ptr)),
                self.len * @sizeOf(ValueT),
                cudaMemcpyDeviceToHost,
            ));
        }
        pub fn view(self: *const Self, comptime Rank: usize, shape: [Rank]usize) TensorView(ValueT, Rank) {
            return TensorView(ValueT, Rank).init(self, shape);
        }
    };
}
