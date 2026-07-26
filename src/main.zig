const std = @import("std");
const stbiw = @import("stb_image_write");
const rl = @import("raylib");
const al = @import("core/linalg.zig");
const mu = @import("microui");
const gui = @import("gui.zig");

const cu = @import("gpu/cuda.zig");
const core = @import("core/core.zig");
const gpu = @import("gpu/gpu.zig");
const sim = @import("sim/sim.zig");

const rc = @import("gpu/raycast.zig");

const RNG_SEED: i32 = 1234;

const AtomicUsize = std.atomic.Value(usize);
const AtomicBool = std.atomic.Value(bool);

const SIMULATION_FRAMERATE: f32 = 144.0;
const RENDERING_FRAMERATE: f32 = 144.0;

const NUM_SPHERES = 4;

const RAY_MAX_DEPTH = 8;
const SAMPLES_PER_PIXEL = 24;
const BLAS_MAX_DEPTH = 24;
const TLAS_MAX_DEPTH = 4;

const PREVIEW_RAY_MAX_DEPTH = 4;
const PREVIEW_SAMPLES_PER_PIXEL = 2;

pub fn setup_teapot_scene(
    world: *core.World,
    scene_scale: al.Vec3,
) !void {
    const mat_glass_id = try world.register_material(.{
        .kind = rc.MaterialKind.Dielectric,
        .albedo = .{ .x = 0.5, .y = 0.5, .z = 0.6 },
        .refractive_index = 10.0,
    });

    const mat_base_id = try world.register_material(.{
        .kind = rc.MaterialKind.Lambertian,
        .albedo = .{ .x = 0.1, .y = 0.1, .z = 0.1 },
    });

    const mat_light_id = try world.register_material(.{
        .kind = rc.MaterialKind.Emissive,
        .emit = .{ .x = 0.95 * 5.0, .y = 0.3 * 5.0, .z = 0.3 * 5.0 }, // yellowish
    });

    //const base_cube = "assets/meshes/cube.txt";
    const base_teapot = "assets/meshes/teapot.txt";

    const s = scene_scale;

    const instances = [_]struct {
        name: []const u8,
        scale: al.Vec3,
        translate: al.Vec3,
        mat: c_uint,
    }{
        // props
        .{ .name = base_teapot, .scale = al.Vec3.full(0.7), .translate = al.Vec3.new(0.0 * s.x, -0.5 * s.y, 0.0 * s.z), .mat = mat_glass_id },
    };

    for (instances) |desc| {
        var mesh = try world.mesh_atlas.parse_mesh_from_file(desc.name);
        _ = al.mat4_scale(&mesh.model, desc.scale);
        _ = al.mat4_translate(&mesh.model, desc.translate);
        mesh.inv_model = al.mat4_inverse(mesh.model).?;
        mesh.material_idx = desc.mat;
    }

    // mock sphere
    try world.spheres.append(.{
        .center = .{.x=-0.5 * s.x, .y=4.0 * s.y, .z=0.0 * s.z},
        .radius = 1.0,
        .material_idx = mat_light_id,
    });

    try world.spheres.append(.{
        .center = .{.x=0.0 * s.x, .y=-101.0 * s.y, .z=0.0 * s.z},
        .radius = 100.0,
        .material_idx = mat_base_id,
    });
}

pub fn setup_box_scene(
    world: *core.World,
    scene_scale: al.Vec3, // now a vector
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
        .{ .name = base_cube, .scale = al.Vec3.new(0.1 * s.x, 1.0 * s.y, 1.0 * s.z), .translate = al.Vec3.new(-s.x, 0.0, 0.0), .mat = mat_red_id },
        .{ .name = base_cube, .scale = al.Vec3.new(0.1 * s.x, 1.0 * s.y, 1.0 * s.z), .translate = al.Vec3.new(s.x, 0.0, 0.0), .mat = mat_green_id },
        .{ .name = base_cube, .scale = al.Vec3.new(1.0 * s.x, 0.1 * s.y, 1.0 * s.z), .translate = al.Vec3.new(0.0, -s.y, 0.0), .mat = mat_gray_id },
        .{ .name = base_cube, .scale = al.Vec3.new(1.0 * s.x, 0.1 * s.y, 1.0 * s.z), .translate = al.Vec3.new(0.0, s.y, 0.0), .mat = mat_gray_id },
        .{ .name = base_cube, .scale = al.Vec3.new(1.0 * s.x, 1.0 * s.y, 0.1 * s.z), .translate = al.Vec3.new(0.0, 0.0, -s.z), .mat = mat_gray_id },
        .{ .name = base_cube, .scale = al.Vec3.new(1.0 * s.x, 1.0 * s.y, 0.1 * s.z), .translate = al.Vec3.new(0.0, 0.0, s.z), .mat = mat_gray_id },

        // light
        .{ .name = base_cube, .scale = al.Vec3.new(0.2 * s.x, 0.1 * s.y, 0.2 * s.z), .translate = al.Vec3.new(0.0, 0.9 * s.y, 0.0), .mat = mat_light_id },

        // props
        .{ .name = base_cube, .scale = al.Vec3.full(0.7), .translate = al.Vec3.new(0.3 * s.x, -0.7 * s.y, -0.5 * s.z), .mat = mat_metal_id },
        .{ .name = base_ico, .scale = al.Vec3.full(1.0), .translate = al.Vec3.new(-0.0 * s.x, -0.0 * s.y, -0.0 * s.z), .mat = mat_metal_id },
    };

    for (instances) |desc| {
        var mesh = try world.mesh_atlas.parse_mesh_from_file(desc.name);
        _ = al.mat4_scale(&mesh.model, desc.scale);
        _ = al.mat4_translate(&mesh.model, desc.translate);
        mesh.inv_model = al.mat4_inverse(mesh.model).?;
        mesh.material_idx = desc.mat;
    }

    // mock sphere
    try world.spheres.append(.{
        .center = .{ .x = -0.0 * s.x, .y = -50 * s.y, .z = 0.2 * s.z },
        .radius = 0.5,
        .material_idx = mat_glass_outer_id,
    });
}

// --- Debug AABB overlay helpers ---

// Project a world-space point to screen coordinates using the same camera
// convention as the CUDA ray generation. Returns null if the point is behind
// the camera.
fn project_point(proj: al.Mat4, world_to_cam: al.Mat4, w: f32, h: f32, p: al.Vec3) ?rl.Vector2 {
    const cam = al.transform_pos(world_to_cam, p); // rigid transform, w == 1
    const clip_x = proj[0][0] * cam.x + proj[0][1] * cam.y + proj[0][2] * cam.z + proj[0][3];
    const clip_y = proj[1][0] * cam.x + proj[1][1] * cam.y + proj[1][2] * cam.z + proj[1][3];
    const clip_w = proj[3][0] * cam.x + proj[3][1] * cam.y + proj[3][2] * cam.z + proj[3][3];
    if (clip_w <= 1e-6) return null; // at or behind the camera plane
    const ndc_x = clip_x / clip_w;
    const ndc_y = clip_y / clip_w;
    return rl.Vector2{
        .x = (ndc_x + 1.0) * 0.5 * w,
        .y = (1.0 - ndc_y) * 0.5 * h,
    };
}

// Draw the 12 edges of an axis-aligned box (given in the space of `model`).
fn draw_wire_box(proj: al.Mat4, world_to_cam: al.Mat4, w: f32, h: f32, model: al.Mat4, bmin: al.Vec3, bmax: al.Vec3, color: rl.Color) void {
    const corners = [8]al.Vec3{
        .{ .x = bmin.x, .y = bmin.y, .z = bmin.z },
        .{ .x = bmax.x, .y = bmin.y, .z = bmin.z },
        .{ .x = bmax.x, .y = bmax.y, .z = bmin.z },
        .{ .x = bmin.x, .y = bmax.y, .z = bmin.z },
        .{ .x = bmin.x, .y = bmin.y, .z = bmax.z },
        .{ .x = bmax.x, .y = bmin.y, .z = bmax.z },
        .{ .x = bmax.x, .y = bmax.y, .z = bmax.z },
        .{ .x = bmin.x, .y = bmax.y, .z = bmax.z },
    };
    var pts: [8]?rl.Vector2 = undefined;
    for (corners, 0..) |c, i| {
        pts[i] = project_point(proj, world_to_cam, w, h, al.transform_pos(model, c));
    }
    const edges = [_][2]usize{
        .{ 0, 1 }, .{ 1, 2 }, .{ 2, 3 }, .{ 3, 0 }, // near face
        .{ 4, 5 }, .{ 5, 6 }, .{ 6, 7 }, .{ 7, 4 }, // far face
        .{ 0, 4 }, .{ 1, 5 }, .{ 2, 6 }, .{ 3, 7 }, // connecting edges
    };
    for (edges) |e| {
        const a = pts[e[0]] orelse continue;
        const b = pts[e[1]] orelse continue;
        rl.DrawLineV(a, b, color);
    }
}

pub fn main() !void {
    var gpa = std.heap.page_allocator;
    var ui = gui.Gui.init();

    const aspect_ratio = 16.0 / 9.0;
    const image_width: u32 = 1280;
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
    // try setup_box_scene(&world, al.Vec3.new(4.0, 3.0, 10.0));
    try setup_teapot_scene(&world, al.Vec3.full(1.0));

    try world.blas.build(&world.mesh_atlas, BLAS_MAX_DEPTH);
    try world.tlas.build(&world.mesh_atlas, TLAS_MAX_DEPTH);

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
        world.blas.nodes.items.len,
        world.tlas.nodes.items.len,
    );
    defer world_dev.deinit();

    try world_dev.spheres.fromHost(world.spheres.items);
    try world_dev.vb.fromHost(&world.mesh_atlas.vb);
    try world_dev.indices.fromHost(world.mesh_atlas.indices.items);
    try world_dev.mesh_ids.fromHost(world.mesh_atlas.mesh_ids.items);
    try world_dev.meshes.fromHost(world.mesh_atlas.meshes.items);
    try world_dev.materials.fromHost(world.materials.items);
    try world_dev.bvh_to_device(&world.blas, &world.tlas, gpa);

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
            .samples_per_pixel = PREVIEW_SAMPLES_PER_PIXEL,
            .temporal_averaging = false,
            .max_depth = PREVIEW_RAY_MAX_DEPTH,
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

    // Debug overlay state.
    var ui_enabled = true;
    rl.EnableCursor();
    var draw_aabb: c_int = 0;

    var blas_draw_depth: u32 = 6;
    var sim_fps: f64 = 0;
    var last_sim_frames: usize = 0;
    var last_fps_time: f64 = rl.GetTime();

    while (!rl.WindowShouldClose()) {
        const idx = shared.ready_idx.load(.acquire);
        const buf = shared.frame_buffers_host[idx];

        rl.UpdateTexture(texture, buf.ptr);

        if (!shared.cam.temporal_averaging) {
            if (!ui_enabled) {
                const mouseDelta = rl.GetMouseDelta();
                camera.yaw += mouseDelta.x * camera.mouse_sensitivity;
                camera.pitch -= mouseDelta.y * camera.mouse_sensitivity;
                camera.update();
                if (rl.IsKeyDown(rl.KEY_W)) camera.move(.Forward);
                if (rl.IsKeyDown(rl.KEY_S)) camera.move(.Back);
                if (rl.IsKeyDown(rl.KEY_A)) camera.move(.Left);
                if (rl.IsKeyDown(rl.KEY_D)) camera.move(.Right);
            }
        }

        // Debug overlay controls.
        if (rl.IsKeyPressed(rl.KEY_Q)) {
            ui_enabled = !ui_enabled;
            if (ui_enabled) {
                rl.EnableCursor();
            } else {
                rl.DisableCursor();
            }
        }
        if (rl.IsKeyPressed(rl.KEY_RIGHT_BRACKET) and blas_draw_depth <= BLAS_MAX_DEPTH) blas_draw_depth += 1;
        if (rl.IsKeyPressed(rl.KEY_LEFT_BRACKET) and blas_draw_depth > 0) blas_draw_depth -= 1;

        if (rl.IsKeyPressed(rl.KEY_P)) {
            rc.launch_clear_buffer(try shared.frame_buffer_dev_accum.view(3, .{ shared.cam.image_height, shared.cam.image_width, 3 }));
            shared.cam.temporal_averaging = !shared.cam.temporal_averaging;
            if (shared.cam.temporal_averaging) {
                shared.cam.max_depth = RAY_MAX_DEPTH;
                shared.cam.samples_per_pixel = SAMPLES_PER_PIXEL;
            } else {
                shared.cam.max_depth = PREVIEW_RAY_MAX_DEPTH;
                shared.cam.samples_per_pixel = PREVIEW_SAMPLES_PER_PIXEL;
            }

            shared.frame_idx = 0;
        }
        shared.cam.camera_to_world = camera.camera_to_world();

        // Sample the simulation frame rate roughly twice a second.
        const now_t = rl.GetTime();
        if (now_t - last_fps_time >= 0.5) {
            const cur = shared.sim_frames.load(.monotonic);
            sim_fps = @as(f64, @floatFromInt(cur - last_sim_frames)) / (now_t - last_fps_time);
            last_sim_frames = cur;
            last_fps_time = now_t;
        }

        rl.BeginDrawing();
        rl.ClearBackground(rl.RAYWHITE);
        rl.DrawTexture(texture, 0, 0, rl.WHITE);

        if (draw_aabb == 1) {
            const fw: f32 = @floatFromInt(image_width);
            const fh: f32 = @floatFromInt(image_height);
            const c2w = camera.camera_to_world();
            if (al.mat4_inverse(c2w)) |w2c| {
                // TLAS instance boxes (already in world space) in green.
                for (world.tlas.nodes.items) |node| {
                    draw_wire_box(camera.proj, w2c, fw, fh, al.mat4_ident(), node.box.min, node.box.max, rl.GREEN);
                }
                // Per-mesh BLAS boxes (model space -> transformed by the mesh
                // model matrix), limited to blas_draw_depth, in red.
                for (world.blas.meshinfo.items, 0..) |info, mesh_i| {
                    const model = world.mesh_atlas.meshes.items[mesh_i].model;
                    const start: usize = @intCast(info.node_offset);
                    const end: usize = start + @as(usize, @intCast(info.node_count));
                    for (world.blas.nodes.items[start..end]) |node| {
                        if (node.depth > blas_draw_depth) continue;
                        draw_wire_box(camera.proj, w2c, fw, fh, model, node.box.min, node.box.max, rl.RED);
                    }
                }
            }
        }

        if (ui_enabled) {
            const render_fps = rl.GetFPS();
            const intersections = shared.last_intersections.load(.monotonic);
            var buf_txt: [160]u8 = undefined;
            const txt = std.fmt.bufPrintSentinel(&buf_txt, "render {d} fps | sim {d:.1} fps | tests {d:.2} M", .{
                render_fps,
                sim_fps,
                @as(f64, @floatFromInt(intersections)) / 1.0e6,
            }, 0) catch unreachable;

            ui.handleInput();
            mu.mu_begin(&ui.ctx);
            if (mu.mu_begin_window(&ui.ctx, "Debug", mu.mu_rect(10, 10, 400, 160)) != 0) {
                mu.mu_layout_row(&ui.ctx, 1, &[_]c_int{-1}, 0);
                mu.mu_label(&ui.ctx, txt.ptr);
                _ = mu.mu_checkbox(&ui.ctx, "Draw TLAS and BLAS", &draw_aabb);
                mu.mu_end_window(&ui.ctx);
            }
            mu.mu_end(&ui.ctx);
            ui.render();
        }

        rl.EndDrawing();
    }

    shared.running.store(false, .release);
    try simulator.stop();
}

