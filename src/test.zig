const std = @import("std");

test "all modules compile" {
    const cu = @import("gpu/cuda.zig");
    const kern = @import("gpu/kernels.zig");
    _ = cu;
    _ = kern;
}
