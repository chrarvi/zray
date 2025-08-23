const std = @import("std");
const stbiw = @import("stb_image_write");

extern fn launch_raycast(img: [*]u8, cam: *const CameraData) void;

// Mirrored in cuda
const CameraData = extern struct {
    image_width: u32,
    image_height: u32,
    focal_length: f32,
    samples_per_pixel: u32,
    max_depth: i32,
};

pub fn main() !void {
    var allocator = std.heap.page_allocator;

    const aspect_ratio = 16.0 / 9.0;
    const image_width = 600;
    const image_height = @max(@divFloor(image_width, aspect_ratio), 1);
    const img = try allocator.alloc(u8, image_width * image_height * 3);
    defer allocator.free(img);

    const camera_data = CameraData{
        .image_width = image_width,
        .image_height = image_height,
        .focal_length = 1.0,
        .samples_per_pixel = 500,
        .max_depth = 100,
    };

    launch_raycast(img.ptr, &camera_data);

    if (stbiw.stbi_write_png("output.png", image_width, image_height, 3, img.ptr, image_width * 3) == 0) {
        return error.ImageWriteFailed;
    }
}
