const std = @import("std");
const stbiw = @import("stb_image_write");
const rl = @import("raylib");
const al = @import("math.zig");
const cu = @import("cuda.zig");

const vb = @import("vb.zig");

const cam = @import("camera.zig");

const rc = @cImport(@cInclude("raycast.h"));

const RNG_SEED: i32 = 1234;

const AtomicUsize = std.atomic.Value(usize);
const AtomicBool = std.atomic.Value(bool);

const World = struct {
    spheres_host: std.ArrayList(rc.Sphere),
    spheres_dev: cu.CudaBuffer(rc.Sphere),
    vb_host: vb.HostVertexBuffer,
    vb_dev: vb.DeviceVertexBuffer,

    fn init(allocator: std.mem.Allocator, spheres_capacity: usize, vertex_capacity: usize) !World {
        return .{
            .spheres_host = std.ArrayList(rc.Sphere).init(allocator),
            .spheres_dev = try cu.CudaBuffer(rc.Sphere).init(spheres_capacity),
            .vb_host = vb.HostVertexBuffer.init(allocator),
            .vb_dev = try vb.DeviceVertexBuffer.init(vertex_capacity),
        };
    }

    fn deinit(self: *World) void {
        self.spheres_host.deinit();
        self.spheres_dev.deinit();
        self.vb_host.deinit();
        self.vb_dev.deinit();
    }
};

const SimSharedState = struct {
    frame_buffers_host: [2][]u8, // double-buffering
    frame_buffer_dev: cu.CudaBuffer(u8),
    ready_idx: AtomicUsize, // which buffer is ready for display
    running: AtomicBool, // shutdown flag
    cam: rc.CameraData,
    world: World,
};

const SIMULATION_FRAMERATE: f32 = 15.0;
const RENDERING_FRAMERATE: f32 = 60.0;

pub extern fn launch_raycast(
    d_img: cu.TensorView(u8, 3),
    cam: *rc.CameraData,
    d_spheres: cu.TensorView(rc.Sphere, 1),
    d_vb_pos: cu.TensorView(f32, 2),
    d_vb_color: cu.TensorView(f32, 2),
    d_vb_normal: cu.TensorView(f32, 2),
) void;

pub fn fill_world(world: *World) !void {
    const metal_mat = rc.Material{
        .kind = rc.MAT_METAL,
        .albedo = .{ .x = 0.8, .y = 0.8, .z = 0.8 },
        .fuzz = 0.05,
    };
    const ground_mat = rc.Material{
        .kind = rc.MAT_LAMBERTIAN,
        .albedo = .{ .x = 0.8, .y = 0.8, .z = 0.8 },
    };

    const world_width = 50.0;
    const world_depth = 50.0;
    const world_height = 0.0;
    const rad_max = 2.0;

    var prng = std.Random.DefaultPrng.init(123456);
    var rand = prng.random();
    const num_spheres = 99;
    for (0..num_spheres) |_| {
        const x = (rand.float(f32) - 0.5) * world_width;
        const z = (rand.float(f32) - 0.5) * world_depth;
        const r = rand.float(f32) * rad_max;
        const y = (rand.float(f32) - 0.5) * world_height + r / 2.0;
        const mat_idx = rand.intRangeAtMost(usize, 0, 2);
        var mat = metal_mat;
        if (mat_idx == 0) {
            mat = rc.Material{
                .kind = rc.MAT_LAMBERTIAN,
                .albedo = .{
                    .x = rand.float(f32),
                    .y = rand.float(f32),
                    .z = rand.float(f32),
                },
            };
        } else if (mat_idx == 1) {
            mat = rc.Material{
                .kind = rc.MAT_METAL,
                .albedo = .{
                    .x = 0.5 + 0.5 * rand.float(f32),
                    .y = 0.5 + 0.5 * rand.float(f32),
                    .z = 0.5 + 0.5 * rand.float(f32),
                },
                .fuzz = rand.float(f32) * 0.5,
            };
        } else if (mat_idx == 2) {
            mat = rc.Material{
                .kind = rc.MAT_EMISSIVE,
                .emit = .{
                    .x = rand.float(f32),
                    .y = rand.float(f32),
                    .z = rand.float(f32),
                },
            };
        }

        try world.spheres_host.append(.{ .center = .{ .x = x, .y = y, .z = z }, .radius = r, .material = mat });
    }
    try world.spheres_host.append(.{ .center = .{ .x = 0.0, .y = -1000.5, .z = -1.0 }, .radius = 1000.0, .material = ground_mat });

    try world.spheres_dev.fromHost(world.spheres_host.items);

    try world.vb_host.push_vertex(.{ -0.3, -0.1, -1.0 }, .{ 1.0, 0.0, 0.0 }, .{ 0.0, 0.0, 1.0 });
    try world.vb_host.push_vertex(.{ 0.0, 0.2, -1.5 }, .{ 1.0, 0.0, 0.0 }, .{ 0.0, 0.0, 1.0 });
    try world.vb_host.push_vertex(.{ 0.3, -0.1, -1.0 }, .{ 1.0, 0.0, 0.0 }, .{ 0.0, 0.0, 1.0 });

    try world.vb_host.push_vertex(.{ 0.6, -0.1, -1.0 }, .{ 0.0, 1.0, 0.0 }, .{ 0.0, 0.0, 1.0 });
    try world.vb_host.push_vertex(.{ 1.0, 0.2, -1.5 }, .{ 0.0, 1.0, 0.0 }, .{ 0.0, 0.0, 1.0 });
    try world.vb_host.push_vertex(.{ 1.3, -0.1, -1.0 }, .{ 0.0, 1.0, 0.0 }, .{ 0.0, 0.0, 1.0 });

    try world.vb_dev.fromHost(&world.vb_host);
}

fn run_sim(shared: *SimSharedState) !void {
    try fill_world(&shared.world);

    var frame: f32 = 0.0;
    const sim_dt = 1.0 / SIMULATION_FRAMERATE;
    var last = rl.GetTime();

    rc.rng_init(shared.cam.image_height, shared.cam.image_width, RNG_SEED);
    defer rc.rng_deinit();

    while (shared.running.load(.acquire)) {
        const now = rl.GetTime();
        if (now - last < sim_dt) {
            std.time.sleep(1_000_000); // 1ms to avoid busy wait
            continue;
        }
        last = now;

        const current_ready = shared.ready_idx.load(.acquire);
        const write_idx: usize = 1 - current_ready;

        launch_raycast(
            try shared.frame_buffer_dev.view(3, .{ shared.cam.image_height, shared.cam.image_width, 3 }),
            &shared.cam,
            try shared.world.spheres_dev.view(1, .{shared.world.spheres_dev.len}),
            try shared.world.vb_dev.pos_buf.view(2, .{ shared.world.vb_dev.pos_buf.len / 3, 3 }),
            try shared.world.vb_dev.color_buf.view(2, .{ shared.world.vb_dev.color_buf.len / 3, 3 }),
            try shared.world.vb_dev.normal_buf.view(2, .{ shared.world.vb_dev.normal_buf.len / 3, 3 }),
        );
        try shared.frame_buffer_dev.toHost(shared.frame_buffers_host[write_idx]);

        shared.ready_idx.store(write_idx, .release);

        frame += 1.0;
    }
}

pub fn main() !void {
    var gpa = std.heap.page_allocator;

    const aspect_ratio = 16.0 / 9.0;
    const image_width: u32 = 1080;
    const image_height: u32 = @intFromFloat(@max(@divFloor(@as(f32, @floatFromInt(image_width)), aspect_ratio), 1));

    // double-buffering
    const buf_size = image_width * image_height * 3;
    const img_host0 = try gpa.alloc(u8, buf_size);
    defer gpa.free(img_host0);
    const img_host1 = try gpa.alloc(u8, buf_size);
    defer gpa.free(img_host1);

    rl.InitWindow(@as(i32, @intCast(image_width)), @as(i32, @intCast(image_height)), "Raytracing demo");
    defer rl.CloseWindow();

    const image = rl.Image{
        .data = img_host0.ptr,
        .width = @as(i32, @intCast(image_width)),
        .height = @as(i32, @intCast(image_height)),
        .mipmaps = 1,
        .format = rl.PIXELFORMAT_UNCOMPRESSED_R8G8B8,
    };

    const texture = rl.LoadTextureFromImage(image);
    defer rl.UnloadTexture(texture);

    var camera = cam.Camera.init_default(image_width, image_height);

    var shared = SimSharedState{
        .frame_buffers_host = .{ img_host0, img_host1 },
        .frame_buffer_dev = try cu.CudaBuffer(u8).init(buf_size),
        .ready_idx = AtomicUsize.init(0),
        .running = AtomicBool.init(true),
        .cam = rc.CameraData{
            .image_width = image_width,
            .image_height = image_height,
            .focal_length = 1.0,
            .samples_per_pixel = 16,
            .max_depth = 10,
            .camera_to_world = camera.camera_to_world(),
            .inv_proj = camera.inv_proj,
        },
        .world = try World.init(gpa, 100, 6 * 3),
    };
    defer shared.world.deinit();
    defer shared.frame_buffer_dev.deinit();

    const sim_thread = try std.Thread.spawn(.{}, run_sim, .{&shared});
    defer sim_thread.join();

    rl.SetTargetFPS(RENDERING_FRAMERATE);
    rl.DisableCursor();

    while (!rl.WindowShouldClose()) {
        const idx = shared.ready_idx.load(.acquire);
        const buf = shared.frame_buffers_host[idx];

        rl.UpdateTexture(texture, buf.ptr);

        const mouseDelta = rl.GetMouseDelta();
        camera.yaw += mouseDelta.x * camera.mouse_sensitivity;
        camera.pitch -= mouseDelta.y * camera.mouse_sensitivity;
        camera.update();

        if (rl.IsKeyDown(rl.KEY_W)) camera.move(.Forward);
        if (rl.IsKeyDown(rl.KEY_S)) camera.move(.Back);
        if (rl.IsKeyDown(rl.KEY_A)) camera.move(.Left);
        if (rl.IsKeyDown(rl.KEY_D)) camera.move(.Right);
        shared.cam.camera_to_world = camera.camera_to_world();

        rl.BeginDrawing();
        rl.ClearBackground(rl.RAYWHITE);
        rl.DrawTexture(texture, 0, 0, rl.WHITE);
        rl.EndDrawing();
    }

    shared.running.store(false, .release);
}
