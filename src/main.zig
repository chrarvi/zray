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

pub fn setup_bvh_scene(
    world: *core.World,
    bvh: *const core.BVHBuilder,
) !void {
    const mat_wire_id = try world.register_material(.{
        .kind = rc.MaterialKind.Lambertian,
        .albedo = .{ .x = 0.8, .y = 0.8, .z = 0.8 },
    });

    for (bvh.nodes.items) |node| {
        if (node.depth == 10) {
            var mesh = try world.mesh_atlas.parse_mesh_from_file("assets/meshes/cube.txt");

            const center = node.box.center();
            const extent = node.box.extent();

            _ = al.mat4_scale(&mesh.model, extent.divc(2.0));
            _ = al.mat4_translate(&mesh.model, center);

            mesh.material_idx = mat_wire_id;
        }
    }
}

pub fn setup_teapot_scene(
    world: *core.World,
    scene_scale: al.Vec3,
) !void {
    const mat_red_id = try world.register_material(.{
        .kind = rc.MaterialKind.Lambertian,
        .albedo = .{ .x = 0.8, .y = 0.0, .z = 0.0 },
    });
    const mat_gray_id = try world.register_material(.{
        .kind = rc.MaterialKind.Lambertian,
        .albedo = .{ .x = 0.5, .y = 0.5, .z = 0.5 },
    });
    const mat_green_id = try world.register_material(.{
        .kind = rc.MaterialKind.Lambertian,
        .albedo = .{ .x = 0.0, .y = 0.8, .z = 0.5 },
    });
    const mat_light_id = try world.register_material(.{
        .kind = rc.MaterialKind.Emissive,
        .emit = .{ .x = 0.95 * 10.0, .y = 0.7 * 10.0, .z = 0.7 * 10.0 }, // yellowish
    });

    const base_cube = "assets/meshes/cube.txt";
    const base_teapot = "assets/meshes/teapot.txt";

    const s = scene_scale;

    const instances = [_]struct {
        name: []const u8,
        scale: al.Vec3,
        translate: al.Vec3,
        mat: c_uint,
    }{
        // props
        .{ .name = base_teapot, .scale = al.Vec3.full(0.7), .translate = al.Vec3.new(0.0 * s.x, -0.5 * s.y, 0.0 * s.z), .mat = mat_gray_id },

        // walls
        .{ .name = base_cube, .scale = al.Vec3.new(0.1 * s.x, 1.0 * s.y, 1.0 * s.z), .translate = al.Vec3.new(-s.x, 0.0, 0.0), .mat = mat_red_id },
        .{ .name = base_cube, .scale = al.Vec3.new(0.1 * s.x, 1.0 * s.y, 1.0 * s.z), .translate = al.Vec3.new(s.x, 0.0, 0.0), .mat = mat_green_id },
        .{ .name = base_cube, .scale = al.Vec3.new(1.0 * s.x, 0.1 * s.y, 1.0 * s.z), .translate = al.Vec3.new(0.0, -s.y, 0.0), .mat = mat_gray_id },
        .{ .name = base_cube, .scale = al.Vec3.new(1.0 * s.x, 0.1 * s.y, 1.0 * s.z), .translate = al.Vec3.new(0.0, s.y, 0.0), .mat = mat_gray_id },
        .{ .name = base_cube, .scale = al.Vec3.new(1.0 * s.x, 1.0 * s.y, 0.1 * s.z), .translate = al.Vec3.new(0.0, 0.0, -s.z), .mat = mat_gray_id },
        .{ .name = base_cube, .scale = al.Vec3.new(1.0 * s.x, 1.0 * s.y, 0.1 * s.z), .translate = al.Vec3.new(0.0, 0.0, s.z), .mat = mat_gray_id },

        // light
        .{ .name = base_cube, .scale = al.Vec3.new(0.2 * s.x, 0.1 * s.y, 0.2 * s.z), .translate = al.Vec3.new(0.0, 0.9 * s.y, 0.0), .mat = mat_light_id },
    };

    for (instances) |desc| {
        var mesh = try world.mesh_atlas.parse_mesh_from_file(desc.name);
        _ = al.mat4_scale(&mesh.model, desc.scale);
        _ = al.mat4_translate(&mesh.model, desc.translate);
        mesh.material_idx = desc.mat;
    }

    // mock sphere
    try world.spheres.append(.{
        .center = .{ .x = 1000.0 * s.x, .y = 0.0 * s.y, .z = 0.0 * s.z },
        .radius = 0.1,
        .material_idx = mat_gray_id,
    });
}

pub fn setup_box_scene(
    world: *core.World,
    scene_scale: al.Vec3, // now a vector
) !void {
    // const mat_red_id = try world.register_material(.{
    //     .kind = rc.MaterialKind.Lambertian,
    //     .albedo = .{ .x = 0.8, .y = 0.0, .z = 0.0 },
    // });
    // const mat_gray_id = try world.register_material(.{
    //     .kind = rc.MaterialKind.Lambertian,
    //     .albedo = .{ .x = 0.5, .y = 0.5, .z = 0.5 },
    // });
    // const mat_green_id = try world.register_material(.{
    //     .kind = rc.MaterialKind.Lambertian,
    //     .albedo = .{ .x = 0.0, .y = 0.8, .z = 0.5 },
    // });
    // const mat_light_id = try world.register_material(.{
    //     .kind = rc.MaterialKind.Emissive,
    //     .emit = .{ .x = 0.95 * 10.0, .y = 0.7 * 10.0, .z = 0.7 * 10.0 }, // yellowish
    // });

    const mat_metal_id = try world.register_material(.{
        .kind = rc.MaterialKind.Metal,
        .albedo = .{ .x = 0.6, .y = 0.6, .z = 0.6 },
        .fuzz = 0.3,
    });

    const mat_glass_outer_id = try world.register_material(.{
        .kind = rc.MaterialKind.Dielectric,
        .albedo = .{ .x = 1.0, .y = 0.0, .z = 0.0 },
        .refractive_index = 1.5,
    });
    // const mat_glass_inner_id = try world.register_material(.{
    //     .kind = rc.MaterialKind.Dielectric,
    //     .albedo = .{ .x = 1.0, .y = 1.0, .z = 1.0 },
    //     .refractive_index = 1.0,
    // });

    // const base_cube = "assets/meshes/cube.txt";
    const base_ico = "assets/meshes/teapot.txt";

    const s = scene_scale; // shorthand

    const instances = [_]struct {
        name: []const u8,
        scale: al.Vec3,
        translate: al.Vec3,
        mat: c_uint,
    }{
        // walls
        // .{ .name = base_cube, .scale = al.Vec3.new(0.1 * s.x, 1.0 * s.y, 1.0 * s.z), .translate = al.Vec3.new(-s.x, 0.0, 0.0), .mat = mat_red_id },
        // .{ .name = base_cube, .scale = al.Vec3.new(0.1 * s.x, 1.0 * s.y, 1.0 * s.z), .translate = al.Vec3.new(s.x, 0.0, 0.0), .mat = mat_green_id },
        // .{ .name = base_cube, .scale = al.Vec3.new(1.0 * s.x, 0.1 * s.y, 1.0 * s.z), .translate = al.Vec3.new(0.0, -s.y, 0.0), .mat = mat_gray_id },
        // .{ .name = base_cube, .scale = al.Vec3.new(1.0 * s.x, 0.1 * s.y, 1.0 * s.z), .translate = al.Vec3.new(0.0, s.y, 0.0), .mat = mat_gray_id },
        // .{ .name = base_cube, .scale = al.Vec3.new(1.0 * s.x, 1.0 * s.y, 0.1 * s.z), .translate = al.Vec3.new(0.0, 0.0, -s.z), .mat = mat_gray_id },
        // .{ .name = base_cube, .scale = al.Vec3.new(1.0 * s.x, 1.0 * s.y, 0.1 * s.z), .translate = al.Vec3.new(0.0, 0.0, s.z), .mat = mat_gray_id },

        // // light
        // .{ .name = base_cube, .scale = al.Vec3.new(0.2 * s.x, 0.1 * s.y, 0.2 * s.z), .translate = al.Vec3.new(0.0, 0.9 * s.y, 0.0), .mat = mat_light_id },

        // props
        // .{ .name = base_cube, .scale = al.Vec3.full(0.7), .translate = al.Vec3.new(0.3 * s.x, -0.7 * s.y, -0.5 * s.z), .mat = mat_metal_id },
        .{ .name = base_ico, .scale = al.Vec3.full(1.0), .translate = al.Vec3.new(-0.0 * s.x, -0.0 * s.y, -0.0 * s.z), .mat = mat_metal_id },
    };

    for (instances) |desc| {
        var mesh = try world.mesh_atlas.parse_mesh_from_file(desc.name);
        _ = al.mat4_scale(&mesh.model, desc.scale);
        _ = al.mat4_translate(&mesh.model, desc.translate);
        mesh.material_idx = desc.mat;
    }

    // mock sphere
    try world.spheres.append(.{
        .center = .{ .x = -0.0 * s.x, .y = -50 * s.y, .z = 0.2 * s.z },
        .radius = 0.5,
        .material_idx = mat_glass_outer_id,
    });
}

pub fn main() !void {
    var gpa = std.heap.page_allocator;

    const aspect_ratio = 16.0 / 9.0;
    const image_width: u32 = 256;
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

    var world = try core.World.init(gpa);
    defer world.deinit();
    try setup_box_scene(&world, al.Vec3.full(1.0));
    // try setup_box_scene(&world, al.Vec3.new(4.0, 3.0, 10.0));
    // try setup_teapot_scene(&world, al.Vec3.new(4.0, 3.0, 10.0));

    // var bvh = core.BVHBuilder.init(gpa);
    // try bvh.build(&world.mesh_atlas, 10);
    // try setup_bvh_scene(&world, &bvh);
    const bvh_max_depth = 10;
    try world.bvh.build(&world.mesh_atlas, bvh_max_depth);

    const n_spheres = world.spheres.items.len;
    const n_vertex = world.mesh_atlas.vb.pos_buf.items.len;
    const n_indices = world.mesh_atlas.indices.items.len;
    const n_meshes = world.mesh_atlas.meshes.items.len;
    const n_materials = world.materials.items.len;
    var world_dev = try gpu.DeviceWorld.init(
        n_spheres,
        n_vertex,
        n_indices,
        n_meshes,
        n_materials,
        bvh_max_depth,
    );
    defer world_dev.deinit();

    try world_dev.spheres.fromHost(world.spheres.items);
    try world_dev.vb.fromHost(&world.mesh_atlas.vb);
    try world_dev.indices.fromHost(world.mesh_atlas.indices.items);
    try world_dev.meshes.fromHost(world.mesh_atlas.meshes.items);
    try world_dev.materials.fromHost(world.materials.items);
    try world_dev.bvh_to_device(&world.bvh, gpa);

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
            .temporal_averaging = false,
            .max_depth = 2,
            .camera_to_world = camera.camera_to_world(),
            .inv_proj = camera.inv_proj,
        },
        .world = &world,
        .world_dev = &world_dev,
        .frame_idx = 0,
    };
    defer shared.frame_buffer_dev.deinit();
    defer shared.frame_buffer_dev_accum.deinit();

    rc.rng_init(shared.cam.image_height, shared.cam.image_width, RNG_SEED);
    defer rc.rng_deinit();

    var simulator = sim.Simulator.init(SIMULATION_FRAMERATE, &shared);
    try simulator.start();

    rl.SetTargetFPS(RENDERING_FRAMERATE);
    rl.DisableCursor();

    while (!rl.WindowShouldClose()) {
        const idx = shared.ready_idx.load(.acquire);
        const buf = shared.frame_buffers_host[idx];

        rl.UpdateTexture(texture, buf.ptr);

        if (!shared.cam.temporal_averaging) {
            const mouseDelta = rl.GetMouseDelta();
            camera.yaw += mouseDelta.x * camera.mouse_sensitivity;
            camera.pitch -= mouseDelta.y * camera.mouse_sensitivity;
            camera.update();

            if (rl.IsKeyDown(rl.KEY_W)) camera.move(.Forward);
            if (rl.IsKeyDown(rl.KEY_S)) camera.move(.Back);
            if (rl.IsKeyDown(rl.KEY_A)) camera.move(.Left);
            if (rl.IsKeyDown(rl.KEY_D)) camera.move(.Right);
        }

        if (rl.IsKeyPressed(rl.KEY_P)) {
            rc.launch_clear_buffer(try shared.frame_buffer_dev_accum.view(3, .{ shared.cam.image_height, shared.cam.image_width, 3 }));
            shared.cam.temporal_averaging = !shared.cam.temporal_averaging;
            if (shared.cam.temporal_averaging) {
                shared.cam.max_depth = 16;
                shared.cam.samples_per_pixel = 32;
            } else {
                shared.cam.max_depth = 2;
                shared.cam.samples_per_pixel = 4;
            }

            shared.frame_idx = 0;
        }
        shared.cam.camera_to_world = camera.camera_to_world();

        rl.BeginDrawing();
        rl.ClearBackground(rl.RAYWHITE);
        rl.DrawTexture(texture, 0, 0, rl.WHITE);
        rl.EndDrawing();
    }

    shared.running.store(false, .release);
    try simulator.stop();
}
