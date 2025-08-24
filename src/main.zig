const std = @import("std");
const stbiw = @import("stb_image_write");
const rl = @import("raylib");

const Camera = @import("camera.zig");

extern fn launch_raycast(img: [*]u8, cam: *const CameraData, spheres: [*]Sphere, spheres_count: usize) void;

// Mirrored in cuda
const CameraData = extern struct {
    image_width: u32,
    image_height: u32,
    focal_length: f32,
    samples_per_pixel: u32,
    max_depth: i32,
};

const MaterialKind = enum(u32) {
    LAMBERTIAN = 0,
    METAL = 1,
};

const Material = extern struct {
    kind: MaterialKind,
    albedo: [3]f32,
    fuzz: f32 = 0.0,
};

const Sphere = extern struct {
    center: [3]f32,
    radius: f32,
    material: Material,
};

pub fn main() !void {
    var allocator = std.heap.page_allocator;

    const aspect_ratio = 16.0 / 9.0;
    const image_width = 600;
    const image_height = @max(@divFloor(image_width, aspect_ratio), 1);
    const img = try allocator.alloc(u8, image_width * image_height * 3);
    defer allocator.free(img);

    var camera_data = CameraData{
        .image_width = image_width,
        .image_height = image_height,
        .focal_length = 1.0,
        .samples_per_pixel = 10,
        .max_depth = 10,
    };

    rl.InitWindow(image_width, image_height, "Raylib test");
    defer rl.CloseWindow();

    const image = rl.Image{
        .data = img.ptr,
        .width = image_width,
        .height = image_height,
        .mipmaps = 1,
        .format = rl.PIXELFORMAT_UNCOMPRESSED_R8G8B8,
    };

    const img_texture: rl.Texture2D = rl.LoadTextureFromImage(image);
    defer rl.UnloadTexture(img_texture);

    var spheres = std.ArrayList(Sphere).init(allocator);
    defer spheres.deinit();

    const lambertian_mat = Material{ .kind = MaterialKind.LAMBERTIAN, .albedo = .{ 0.1, 0.2, 0.5 } };
    const metal_mat = Material{ .kind = MaterialKind.METAL, .albedo = .{ 0.8, 0.8, 0.8 }, .fuzz = 0.05 };
    const metal_mat2 = Material{ .kind = MaterialKind.METAL, .albedo = .{ 0.8, 0.6, 0.3 }};
    const ground_mat = Material{ .kind = MaterialKind.LAMBERTIAN, .albedo = .{ 0.8, 0.8, 0.8 } };
    try spheres.append(
        .{ .center = .{ -0.8, -0.15, -0.8 }, .radius = 0.3, .material = lambertian_mat },
    );
    try spheres.append(
        .{ .center = .{ 0.0, 0.0, -1.2 }, .radius = 0.5, .material = metal_mat },
    );
    try spheres.append(
        .{ .center = .{ 0.8, -0.15, -0.8 }, .radius = 0.3, .material = metal_mat2 },
    );
    try spheres.append(.{ .center = .{ 0.0, -100.5, -1.0 }, .radius = 100.0, .material = ground_mat });

    const cam = Camera.init(
        .{0.0, 0.0, 3.0},  // eye
        .{0.0, 0.0, 0.0},  // target
        .{0.0, 1.0, 0.0},  // up
    );

    const view = cam.look_at();
    const cam_to_world = cam.camera_to_world();

    std.debug.print("View matrix: {any}\n", .{view});
    std.debug.print("CameraToWorld matrix: {any}\n", .{cam_to_world});

    rl.SetTargetFPS(15);

    var frame: f32 = 0.0;
    while (!rl.WindowShouldClose()) {
        const t = frame * 0.03;
        spheres.items[0].center[0] = -0.8 + 0.3 * @sin(t);
        spheres.items[1].center[1] = 0.0 + 0.2 * @sin(t * 1.5);
        spheres.items[2].center[0] = 0.8 + 0.3 * @sin(t * 0.8);
        launch_raycast(img.ptr, &camera_data, spheres.items.ptr, spheres.items.len);

        rl.UpdateTexture(img_texture, img.ptr);

        rl.BeginDrawing();
        rl.ClearBackground(rl.RAYWHITE);

        rl.DrawTexturePro(
            img_texture,
            rl.Rectangle{ .x = 0, .y = 0, .width = image_width, .height = image_height },
            rl.Rectangle{ .x = 0, .y = 0, .width = image_width, .height = image_height },
            rl.Vector2{ .x = 0, .y = 0 },
            0,
            rl.WHITE,
        );

        rl.EndDrawing();

        frame += 1.0;
    }
}
