const std = @import("std");
const cu = @import("cuda.zig");

const TensorView = cu.TensorView;

pub extern fn add_f32_2d(a: TensorView(f32, 2), b: TensorView(f32, 2), b: TensorView(f32, 2)) void;
pub extern fn add_i32_2d(a: TensorView(i32, 2), b: TensorView(i32, 2), b: TensorView(i32, 2)) void;

test "add_f32_2d kernel computes correctly" {
    var a_host: [10 * 10]f32 = undefined;
    @memset(&a_host, 2);
    var a = try cu.CudaBuffer(f32).init(10 * 10);
    defer a.deinit();
    try a.fromHost(&a_host);

    var b_host: [10 * 10]f32 = undefined;
    @memset(&b_host, 10);
    var b = try cu.CudaBuffer(f32).init(10 * 10);
    defer b.deinit();
    try b.fromHost(&b_host);

    var c = try cu.CudaBuffer(f32).init(10 * 10);
    defer c.deinit();

    const a_view = a.view(2, .{10, 10});
    const b_view = b.view(2, .{10, 10});
    const c_view = c.view(2, .{10, 10});

    // Call kernel twice like your main
    add_f32_2d(a_view, b_view, c_view);
    add_f32_2d(c_view, b_view, c_view);

    var c_host: [10 * 10]f32 = undefined;
    try c.toHost(&c_host);

    for (0..10) |y| {
        for (0..10) |x| {
            const expected = 2 + 10 + 10;
            try std.testing.expect(c_host[y * 10 + x] == expected);
        }
    }
}
