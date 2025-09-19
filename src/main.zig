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

const SIMULATION_FRAMERATE: f32 = 10.0;
const RENDERING_FRAMERATE: f32 = 30.0;

const NUM_SPHERES = 4;

pub fn setup_box_scene(
    world: *core.World,
    scene_scale: al.Vec3, // now a vector
) !void {
    const mat_red = rc.Material{
        .kind = rc.MaterialKind.Lambertian,
        .albedo = .{ .x = 0.8, .y = 0.0, .z = 0.0 },
    };
    const mat_gray = rc.Material{
        .kind = rc.MaterialKind.Lambertian,
        .albedo = .{ .x = 0.5, .y = 0.5, .z = 0.5 },
    };
    const mat_green = rc.Material{
        .kind = rc.MaterialKind.Lambertian,
        .albedo = .{ .x = 0.0, .y = 0.8, .z = 0.5 },
    };
    const mat_light = rc.Material{
        .kind = rc.MaterialKind.Emissive,
        .emit = .{ .x = 0.95 * 10.0, .y = 0.7 * 10.0, .z = 0.7 * 10.0 }, // yellowish
    };

    const mat_metal = rc.Material{
        .kind = rc.MaterialKind.Metal,
        .albedo = .{.x = 0.6, .y = 0.6, .z = 0.6},
        .fuzz = 0.3,
    };

    const mat_glass_outer = rc.Material{
        .kind = rc.MaterialKind.Dialectric,
        .albedo = .{.x = 1.0, .y = 1.0, .z = 1.0},
        .refractive_index = 1.5,
    };
    const mat_glass_inner = rc.Material{
        .kind = rc.MaterialKind.Dialectric,
        .albedo = .{.x = 1.0, .y = 1.0, .z = 1.0},
        .refractive_index = 1.0 / mat_glass_outer.refractive_index,
    };

    const red = try world.register_material(mat_red);
    const gray = try world.register_material(mat_gray);
    const green = try world.register_material(mat_green);
    const light = try world.register_material(mat_light);
    const metal = try world.register_material(mat_metal);
    const glass_outer = try world.register_material(mat_glass_outer);
    _ = try world.register_material(mat_glass_inner);

    const base_cube = "assets/meshes/cube.txt";
    const base_ico = "assets/meshes/icosahedron.txt";

    const s = scene_scale; // shorthand

    const instances = [_]struct {
        name: []const u8,
        scale: al.Vec3,
        translate: al.Vec3,
        mat: c_uint,
    }{
        // walls
        .{ .name = base_cube, .scale = al.Vec3.new(0.1 * s.x, 1.0 * s.y, 1.0 * s.z), .translate = al.Vec3.new(-s.x, 0.0, 0.0), .mat = red },
        .{ .name = base_cube, .scale = al.Vec3.new(0.1 * s.x, 1.0 * s.y, 1.0 * s.z), .translate = al.Vec3.new(s.x, 0.0, 0.0), .mat = green },
        .{ .name = base_cube, .scale = al.Vec3.new(1.0 * s.x, 0.1 * s.y, 1.0 * s.z), .translate = al.Vec3.new(0.0, -s.y, 0.0), .mat = gray },
        .{ .name = base_cube, .scale = al.Vec3.new(1.0 * s.x, 0.1 * s.y, 1.0 * s.z), .translate = al.Vec3.new(0.0, s.y, 0.0), .mat = gray },
        .{ .name = base_cube, .scale = al.Vec3.new(1.0 * s.x, 1.0 * s.y, 0.1 * s.z), .translate = al.Vec3.new(0.0, 0.0, -s.z), .mat = gray },
        .{ .name = base_cube, .scale = al.Vec3.new(1.0 * s.x, 1.0 * s.y, 0.1 * s.z), .translate = al.Vec3.new(0.0, 0.0, s.z), .mat = gray },

        // light
        .{ .name = base_cube, .scale = al.Vec3.new(0.2 * s.x, 0.1 * s.y, 0.2 * s.z), .translate = al.Vec3.new(0.0, 0.9 * s.y, 0.0), .mat = light },

        // props
        .{ .name = base_cube, .scale = al.Vec3.full(0.7), .translate = al.Vec3.new(0.3 * s.x, -0.7 * s.y, -0.5 * s.z), .mat = metal },
        .{ .name = base_ico, .scale = al.Vec3.full(1.0), .translate = al.Vec3.new(-0.3 * s.x, -0.6 * s.y, -0.2 * s.z), .mat = metal },
    };

    for (instances) |desc| {
        var mesh = try world.mesh_atlas.parse_mesh_from_file(desc.name);
        _ = al.mat4_scale(&mesh.model, desc.scale);
        _ = al.mat4_translate(&mesh.model, desc.translate);
        mesh.material_idx = desc.mat;
    }

    // mock sphere
    try world.spheres.append(.{
        .center = .{ .x = -0.0 * s.x, .y = -0.5 * s.y, .z = 0.2 * s.z },
        .radius = 0.5,
        .material_idx = glass_outer,
    });
    try world.spheres.append(.{
        .center = .{ .x = -0.0 * s.x, .y = -0.5 * s.y, .z = 0.2 * s.z },
        .radius = 0.35,
        .material_idx = red,
    });
}

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
        .frame_buffer_dev_accum = try cu.CudaBuffer(f32).init(buf_size),
        .ready_idx = AtomicUsize.init(0),
        .running = AtomicBool.init(true),
        .cam = rc.CameraData{
            .image_width = image_width,
            .image_height = image_height,
            .focal_length = 1.0,
            .samples_per_pixel = 8,
            .temporal_averaging = true,
            .max_depth = 8,
            .camera_to_world = camera.camera_to_world(),
            .inv_proj = camera.inv_proj,
        },
        .world = try core.World.init(gpa),
        .world_dev = try gpu.DeviceWorld.init(2, (36 * 8 + 60) * 4, (36 * 8 + 60), 9, 7),
    };
    defer shared.world.deinit();
    defer shared.frame_buffer_dev.deinit();
    defer shared.frame_buffer_dev_accum.deinit();

    try setup_box_scene(&shared.world, al.Vec3.new(4.0, 3.0, 10.0));
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
