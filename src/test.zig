const std = @import("std");

test "all modules compile" {
    const cu = @import("cuda.zig");
    const kern = @import("kernels.zig");
    _ = cu;
    _ = kern;
}
