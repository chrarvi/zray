const std = @import("std");
const stbiw = @import("stb_image_write");

extern fn launch_kernel(img: [*]u8, width: c_int, height: c_int) void;

pub fn main() !void {
    const stdout = std.io.getStdOut().writer();
    try stdout.print("Hello world\n", .{});

    const width = 800;
    const height = 800;

    var allocator = std.heap.page_allocator;
    const img = try allocator.alloc(u8, width * height * 3);
    defer allocator.free(img);

    launch_kernel(img.ptr, width, height);

    if (stbiw.stbi_write_png("output.png", width, height, 3, img.ptr, width * 3) == 0) {
        return error.ImageWriteFailed;
    }
}
