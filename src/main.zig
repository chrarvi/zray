const std = @import("std");
const stbiw = @import("stb_image_write");
const rl = @import("raylib");
const al = @import("core/linalg.zig");

const cu = @import("gpu/cuda.zig");
const core = @import("core/core.zig");
const gpu = @import("gpu/gpu.zig");
const sim = @import("sim/sim.zig");

const rc = @import("gpu/raycast.zig");

const RNG_SEED: i32 = 1234;

const AtomicUsize = std.atomic.Value(usize);
const AtomicBool = std.atomic.Value(bool);

const SIMULATION_FRAMERATE: f32 = 30.0;
const RENDERING_FRAMERATE: f32 = 30.0;

const NUM_SPHERES = 4;

pub fn fill_world(world: *core.World) !void {
    const world_width = 50.0;
    const world_depth = 50.0;
    const world_height = 0.0;
    const rad_max = 2.0;

    const wireframe_mat = rc.Material{
        .kind = rc.MaterialKind.Dialectric,
        .refractive_index = 1.5,
        .albedo = .{ .x = 0.5, .y = 0.5, .z = 0.5 },
    };
    try world.materials.append(wireframe_mat);

    var prng = std.Random.DefaultPrng.init(123456);
    var rand = prng.random();
    const num_spheres = NUM_SPHERES;
    for (0..num_spheres) |_| {
        const x = (rand.float(f32) - 0.5) * world_width;
        const z = (rand.float(f32) - 0.5) * world_depth;
        const r = rand.float(f32) * rad_max;
        const y = (rand.float(f32) - 0.5) * world_height + r / 2.0;
        const mat_idx = rand.intRangeAtMost(usize, 0, 2);
        var mat: rc.Material = undefined;
        if (mat_idx == 0) {
            mat = rc.Material{
                .kind = rc.MaterialKind.Dialectric,
                .refractive_index = 1.5,
                .albedo = .{
                    .x = rand.float(f32),
                    .y = rand.float(f32),
                    .z = rand.float(f32),
                },
            };
        } else if (mat_idx == 1) {
            mat = rc.Material{
                .kind = rc.MaterialKind.Metal,
                .albedo = .{
                    .x = 0.5 + 0.5 * rand.float(f32),
                    .y = 0.5 + 0.5 * rand.float(f32),
                    .z = 0.5 + 0.5 * rand.float(f32),
                },
                .fuzz = rand.float(f32) * 0.5,
            };
        } else if (mat_idx == 2) {
            mat = rc.Material{
                .kind = rc.MaterialKind.Emissive,
                .emit = .{
                    .x = rand.float(f32),
                    .y = rand.float(f32),
                    .z = rand.float(f32),
                },
            };
        }

        try world.materials.append(mat);
        try world.spheres.append(.{
            .center = .{ .x = x, .y = y, .z = z },
            .radius = r,
            .material_idx = @as(u32, @intCast(world.materials.items.len - 1)),
        });
    }

    const ground_mat = rc.Material{
        .kind = rc.MaterialKind.Lambertian,
        .albedo = .{ .x = 0.8, .y = 0.8, .z = 0.8 },
    };
    try world.materials.append(ground_mat);

    try world.spheres.append(.{ .center = .{ .x = 0.0, .y = -1000.5, .z = -1.0 }, .radius = 1000.0, .material_idx = @as(u32, @intCast(world.materials.items.len - 1)) });

    var icosa_mesh = try world.mesh_atlas.parse_mesh_from_file("assets/meshes/icosahedron.txt");
    var cube_mesh = try world.mesh_atlas.parse_mesh_from_file("assets/meshes/cube.txt");

    _ = al.mat4_translate(&icosa_mesh.model, al.Vec3.new(1.0, 1.0, 0.0));
    _ = al.mat4_translate(&cube_mesh.model, al.Vec3.new(-1.0, 1.0, 0.0));
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

    var camera = core.Camera.init_default(image_width, image_height);

    var shared = sim.SimSharedState{
        .frame_buffers_host = .{ img_host0, img_host1 },
        .frame_buffer_dev = try cu.CudaBuffer(u8).init(buf_size),
        .ready_idx = AtomicUsize.init(0),
        .running = AtomicBool.init(true),
        .cam = rc.CameraData{
            .image_width = image_width,
            .image_height = image_height,
            .focal_length = 1.0,
            .samples_per_pixel = 16,
            .max_depth = 8,
            .camera_to_world = camera.camera_to_world(),
            .inv_proj = camera.inv_proj,
        },
        .world = try core.World.init(gpa),
        .world_dev = try gpu.DeviceWorld.init(NUM_SPHERES + 1, (36 + 60) * 4, (36 + 60), 2, 6),
    };
    defer shared.world.deinit();
    defer shared.frame_buffer_dev.deinit();

    try fill_world(&shared.world);
    rc.rng_init(shared.cam.image_height, shared.cam.image_width, RNG_SEED);
    defer rc.rng_deinit();

    try shared.world_dev.spheres.fromHost(shared.world.spheres.items);
    try shared.world_dev.vb.fromHost(&shared.world.mesh_atlas.vb);
    try shared.world_dev.indices.fromHost(shared.world.mesh_atlas.indices.items);
    try shared.world_dev.meshes.fromHost(shared.world.mesh_atlas.meshes.items);
    try shared.world_dev.materials.fromHost(shared.world.materials.items);

    var bvh = core.BoundingVolumeHierarchy.init();
    bvh.build(&shared.world.mesh_atlas);
    var simulator = sim.Simulator.init(SIMULATION_FRAMERATE, &shared);
    try simulator.start();

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
    try simulator.stop();
}
