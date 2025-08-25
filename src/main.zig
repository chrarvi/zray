const std = @import("std");
const stbiw = @import("stb_image_write");
const rl = @import("raylib");
const al = @import("math.zig");

const Camera = @import("Camera.zig");
const mat = @import("material.zig");

extern fn launch_raycast(img: [*]u8, cam: *const CameraData, spheres: [*]Sphere, spheres_count: usize) void;

const CameraData = extern struct {
    image_width: u32,
    image_height: u32,
    focal_length: f32,
    samples_per_pixel: u32,
    max_depth: i32,
    camera_to_world: [16]f32,
};

const Sphere = extern struct {
    center: [3]f32,
    radius: f32,
    material: Material,
};

const Material = extern struct {
    kind: u32,
    albedo: [3]f32,
    fuzz: f32,
};

const AtomicUsize = std.atomic.Value(usize);
const AtomicBool = std.atomic.Value(bool);

const SimSharedState = struct {
    buffers: [2][]u8,           // double-buffering
    ready_idx: AtomicUsize,     // which buffer is ready for display
    running: AtomicBool,        // shutdown flag
    cam: *const CameraData,
    spheres: [*]Sphere,
    spheres_count: usize,
};

const SIMULATION_FRAMERATE: f32 = 10.0;
const RENDERING_FRAMERATE: f32 = 60.0;

fn run_sim(shared: *SimSharedState) !void {
    var frame: f32 = 0.0;
    const sim_dt = 1.0 / SIMULATION_FRAMERATE;
    var last = rl.GetTime();

    while (shared.running.load(.acquire)) {
        const now = rl.GetTime();
        if (now - last < sim_dt) {
            std.time.sleep(1_000_000); // 1ms to avoid busy wait
            continue;
        }
        last = now;

        const t = frame * 0.03;
        shared.spheres[0].center[0] = -0.8 + 0.3 * @sin(t);
        shared.spheres[1].center[1] =  0.0 + 0.2 * @sin(t * 1.5);
        shared.spheres[2].center[0] =  0.8 + 0.3 * @sin(t * 0.8);

        const current_ready = shared.ready_idx.load(.acquire);
        const write_idx: usize = 1 - current_ready;

        launch_raycast(shared.buffers[write_idx].ptr, shared.cam, shared.spheres, shared.spheres_count);

        shared.ready_idx.store(write_idx, .release);

        frame += 1.0;
    }
}

pub fn main() !void {
    var gpa = std.heap.page_allocator;

    const aspect_ratio = 16.0 / 9.0;
    const image_width: u32 = 600;
    const image_height: u32 = @intFromFloat(@max(@divFloor(@as(f32, @floatFromInt(image_width)), aspect_ratio), 1));

    // double-buffering
    const buf_size = image_width * image_height * 3;
    const img0 = try gpa.alloc(u8, buf_size);
    defer gpa.free(img0);
    const img1 = try gpa.alloc(u8, buf_size);
    defer gpa.free(img1);

    // camera + spheres (owned by sim thread; main must not mutate)
    var camera_data = CameraData{
        .image_width = image_width,
        .image_height = image_height,
        .focal_length = 1.0,
        .samples_per_pixel = 16,
        .max_depth = 10,
        .camera_to_world = [_]f32{
            1,0,0,0,
            0,1,0,0,
            0,0,1,0,
            0,0,0,1,
        },
    };

    var spheres = try gpa.alloc(Sphere, 4);
    defer gpa.free(spheres);
    spheres[0] = .{ .center = .{ -0.8, -0.15, -0.8 }, .radius = 0.3, .material = .{ .kind = 0, .albedo = .{0.1,0.2,0.5}, .fuzz = 0 } };
    spheres[1] = .{ .center = .{  0.0,  0.0, -1.2 }, .radius = 0.5, .material = .{ .kind = 1, .albedo = .{0.8,0.8,0.8}, .fuzz = 0.05 } };
    spheres[2] = .{ .center = .{  0.8, -0.15, -0.8 }, .radius = 0.3, .material = .{ .kind = 1, .albedo = .{0.8,0.6,0.3}, .fuzz = 0 } };
    spheres[3] = .{ .center = .{  0.0, -100.5, -1.0 }, .radius = 100.0, .material = .{ .kind = 0, .albedo = .{0.8,0.8,0.8}, .fuzz = 0 } };

    rl.InitWindow(@as(i32, @intCast(image_width)), @as(i32, @intCast(image_height)), "Sim/Render decoupled");
    defer rl.CloseWindow();

    const image = rl.Image{
        .data = img0.ptr,
        .width = @as(i32, @intCast(image_width)),
        .height = @as(i32, @intCast(image_height)),
        .mipmaps = 1,
        .format = rl.PIXELFORMAT_UNCOMPRESSED_R8G8B8,
    };

    const texture = rl.LoadTextureFromImage(image);
    defer rl.UnloadTexture(texture);

    var shared = SimSharedState{
        .buffers = .{ img0, img1 },
        .ready_idx = AtomicUsize.init(0),
        .running = AtomicBool.init(true),
        .cam = &camera_data,
        .spheres = spheres.ptr,
        .spheres_count = spheres.len,
    };
    const sim_thread = try std.Thread.spawn(.{}, run_sim, .{ &shared });
    defer sim_thread.join();

    rl.SetTargetFPS(RENDERING_FRAMERATE);

    while (!rl.WindowShouldClose()) {
        const idx = shared.ready_idx.load(.acquire);
        const buf = shared.buffers[idx];

        rl.UpdateTexture(texture, buf.ptr);

        rl.BeginDrawing();
        rl.ClearBackground(rl.RAYWHITE);
        rl.DrawTexture(texture, 0, 0, rl.WHITE);
        rl.EndDrawing();
    }

    shared.running.store(false, .release);
}
