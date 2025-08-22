const std = @import("std");
const stbiw = @import("stb_image_write");

extern fn launch_raycast(img: [*]u8, width: c_int, height: c_int) void;

const vec3f = @Vector(3, f32);

const Ray = struct {
    origin: vec3f,
    dir: vec3f,
};

fn vec3f_norm(vec: vec3f) f32 {
    return @sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
}

fn vec3f_normalize(vec: vec3f) vec3f {
    const norm = vec3f_norm(vec);
    return vec / @as(vec3f, @splat(norm));}

fn ray_color(ray: *const Ray) vec3f {
    const unit_dir = vec3f_normalize(ray.dir);
    const a = 0.5 * (unit_dir[1] + 1.0);
    const a_v = @as(vec3f, @splat(a));
    return @as(vec3f, @splat(1.0 - a)) * @as(vec3f, @splat(1.0)) + a_v*vec3f{0.5, 0.7, 1.0};
}

pub fn main() !void {
    var allocator = std.heap.page_allocator;

    const aspect_ratio = 16.0 / 9.0;
    const image_width = 400;
    const image_height: i32 = @max(@divFloor(image_width, aspect_ratio), 1);
    const img = try allocator.alloc(u8, image_width * image_height * 3);
    defer allocator.free(img);

    const viewport_height: f32 = 2.0;
    const viewport_width = viewport_height * @as(f32, @floatFromInt(image_width)) / @as(f32, @floatFromInt(image_height));

    const focal_length = 1.0;
    const camera_center = vec3f{ 0.0, 0.0, 0.0 };

    const viewport_u = vec3f{ viewport_width, 0.0, 0.0 };
    const viewport_v = vec3f{ 0.0, -viewport_height, 0.0 };

    const pixel_delta_u = viewport_u / @as(vec3f, @splat(@as(f32, @floatFromInt(image_width))));
    const pixel_delta_v = viewport_v / @as(vec3f, @splat(@as(f32, @floatFromInt(image_height))));

    const two = @as(vec3f, @splat(2.0));
    const viewport_upper_left = camera_center - vec3f{ 0.0, 0.0, focal_length } - viewport_u / two - viewport_v / two;
    const pixel00_loc = viewport_upper_left + (pixel_delta_u + pixel_delta_v) / two;

    for (0..image_height) |img_y| {
        for (0..image_width) |img_x| {
            const x_vf32 = @as(vec3f, @splat(@as(f32, @floatFromInt(img_x))));
            const y_vf32 = @as(vec3f, @splat(@as(f32, @floatFromInt(img_y))));
            const pixel_center = pixel00_loc + (x_vf32 * pixel_delta_u) + (y_vf32 * pixel_delta_v);
            const ray_direction = pixel_center - camera_center;
            const ray = Ray{.dir = ray_direction, .origin = camera_center};
            const color = ray_color(&ray);
            const idx = 3 * (image_width * img_y + img_x);

            img[idx+0] = @intFromFloat(255.0 * @max(@min(1.0, color[0]), 0.0));
            img[idx+1] = @intFromFloat(255.0 * @max(@min(1.0, color[1]), 0.0));
            img[idx+2] = @intFromFloat(255.0 * @max(@min(1.0, color[2]), 0.0));
        }
    }

    // launch_raycast(img.ptr, image_width, image_height);

    if (stbiw.stbi_write_png("output.png", image_width, image_height, 3, img.ptr, image_width * 3) == 0) {
        return error.ImageWriteFailed;
    }
}
