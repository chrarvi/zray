const std = @import("std");
const stbiw = @import("stb_image_write");
const rl = @import("raylib");
const al = @import("math.zig");

const Camera = @import("Camera.zig");

const rc = @cImport(@cInclude("raycast.h"));

const RNG_SEED: i32 = 1234;

const AtomicUsize = std.atomic.Value(usize);
const AtomicBool = std.atomic.Value(bool);

const SimSharedState = struct {
    buffers: [2][]u8, // double-buffering
    ready_idx: AtomicUsize, // which buffer is ready for display
    running: AtomicBool, // shutdown flag
    cam: rc.CameraData,
    spheres: [*]rc.Sphere,
    spheres_count: usize,
};

const SIMULATION_FRAMERATE: f32 = 30.0;
const RENDERING_FRAMERATE: f32 = 60.0;

const VertexBuffer = struct {
    const Buf = std.ArrayList(al.Vec3);
    // struct of arrays over array of structs
    p_buf: std.ArrayList(al.Vec3),
    n_buf: std.ArrayList(al.Vec3),
    c_buf: std.ArrayList(al.Vec3),

    count: usize,

    fn init(allocator: std.mem.Allocator) !VertexBuffer {
        return VertexBuffer {
            .p_buf = Buf.init(allocator),
            .n_buf = Buf.init(allocator),
            .c_buf = Buf.init(allocator),
            .count = 0,
        };
    }

    fn deinit(self: *VertexBuffer) void {
        self.p_buf.deinit();
        self.c_buf.deinit();
        self.n_buf.deinit();
    }

    fn push_vertex(self: *VertexBuffer, p: al.Vec3, n: al.Vec3, c: al.Vec3) !void {
        try self.p_buf.append(p);
        try self.c_buf.append(c);
        try self.n_buf.append(n);
        self.count += 1;
    }
};

fn run_sim(shared: *SimSharedState) !void {
    var frame: f32 = 0.0;
    const sim_dt = 1.0 / SIMULATION_FRAMERATE;
    var last = rl.GetTime();

    rc.init_cuda(&shared.cam, shared.spheres_count, RNG_SEED);
    defer rc.cleanup_cuda();

    while (shared.running.load(.acquire)) {
        const now = rl.GetTime();
        if (now - last < sim_dt) {
            std.time.sleep(1_000_000); // 1ms to avoid busy wait
            continue;
        }
        last = now;

        const t = frame * 0.03;
        shared.spheres[0].center.x = -0.8 + 0.3 * @sin(t);
        shared.spheres[1].center.y = 0.0 + 0.2 * @sin(t * 1.5);
        shared.spheres[2].center.x = 0.8 + 0.3 * @sin(t * 0.8);

        const current_ready = shared.ready_idx.load(.acquire);
        const write_idx: usize = 1 - current_ready;

        rc.update_spheres(shared.spheres, shared.spheres_count);
        rc.launch_raycast(shared.buffers[write_idx].ptr, &shared.cam, shared.spheres, shared.spheres_count);

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

    var vb = try VertexBuffer.init(gpa);
    defer vb.deinit();
    try vb.push_vertex(.{1.0, 1.0, 1.0}, .{0.0, 0.0, 0.0}, .{1.0, 0.0, 0.0});
    try vb.push_vertex(.{-1.0, -1.0, 1.0}, .{0.0, 0.0, 0.0}, .{0.0, 1.0, 0.0});
    try vb.push_vertex(.{1.0, -1.0, 1.0}, .{0.0, 0.0, 0.0}, .{0.0, 0.0, 1.0});

    const d_vb = rc.vb_alloc(vb.count);
    defer rc.vb_free(d_vb);

    var spheres = try gpa.alloc(rc.Sphere, 4);
    defer gpa.free(spheres);

    const lambertian_mat = rc.Material{
        .kind = rc.MAT_LAMBERTIAN,
        .albedo = .{ .x = 0.1, .y = 0.2, .z = 0.5 },
    };
    const metal_mat = rc.Material{
        .kind = rc.MAT_METAL,
        .albedo = .{ .x = 0.8, .y = 0.8, .z = 0.8 },
        .fuzz = 0.05,
    };
    const ground_mat = rc.Material{
        .kind = rc.MAT_LAMBERTIAN,
        .albedo = .{ .x = 0.8, .y = 0.8, .z = 0.8 },
    };
    const emit_mat = rc.Material{
        .kind = rc.MAT_EMISSIVE,
        .emit = .{ .x = 1.0, .y = 1.0, .z = 1.0 },
    };
    spheres[0] = .{ .center = .{ .x = -0.8, .y = -0.15, .z = -0.8 }, .radius = 0.3, .material = lambertian_mat };
    spheres[1] = .{ .center = .{ .x = 0.0, .y = 0.0, .z = -1.2 }, .radius = 0.5, .material = emit_mat };
    spheres[2] = .{ .center = .{ .x = 0.8, .y = -0.15, .z = -0.8 }, .radius = 0.3, .material = metal_mat };
    spheres[3] = .{ .center = .{ .x = 0.0, .y = -100.5, .z = -1.0 }, .radius = 100.0, .material = ground_mat };

    rl.InitWindow(@as(i32, @intCast(image_width)), @as(i32, @intCast(image_height)), "Raytracing demo");
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

    const camera = Camera.init(
        .{0.0, 0.0, 0.0},
        .{0.0, 0.0, -1.0},
        .{0.0, 1.0, 0.0},
    );

    var shared = SimSharedState{
        .buffers = .{ img0, img1 },
        .ready_idx = AtomicUsize.init(0),
        .running = AtomicBool.init(true),
        .cam = rc.CameraData{
            .image_width = image_width,
            .image_height = image_height,
            .focal_length = 1.0,
            .samples_per_pixel = 32,
            .max_depth = 10,
            .camera_to_world = camera.camera_to_world(),
        },
        .spheres = spheres.ptr,
        .spheres_count = spheres.len,
    };
    const sim_thread = try std.Thread.spawn(.{}, run_sim, .{&shared});
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
