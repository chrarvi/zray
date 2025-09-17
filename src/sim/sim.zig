const std = @import("std");
const core = @import("../core/core.zig");
const gpu = @import("../gpu/gpu.zig");
const rc = @import("../gpu/raycast.zig");

const AtomicUsize = std.atomic.Value(usize);
const AtomicBool = std.atomic.Value(bool);

pub const SimSharedState = struct {
    frame_buffers_host: [2][]u8, // double-buffering
    frame_buffer_dev: gpu.cuda.CudaBuffer(u8),
    ready_idx: AtomicUsize, // which buffer is ready for display
    running: AtomicBool, // shutdown flag
    cam: rc.CameraData,
    world: core.World,
    world_dev: gpu.DeviceWorld,
};

pub const Simulator = struct {
    frame_rate: f32,
    state: *SimSharedState,
    sim_thread: ?std.Thread = null,

    pub fn init(frame_rate: f32, state: *SimSharedState) Simulator {
        return Simulator{
            .frame_rate = frame_rate,
            .state = state,
        };
    }


    pub fn start(self: *Simulator) !void {
        if (self.sim_thread != null) {
            return error.SimulationAlreadyRunning;
        }
        self.sim_thread = try std.Thread.spawn(.{}, run_sim, .{self.state, self.frame_rate});
    }

    pub fn stop(self: *Simulator) !void {
        if (self.sim_thread == null) {
            return error.SimulationNotRunning;
        }
        self.sim_thread.?.join();
        self.sim_thread = null;
    }

};

fn run_sim(shared: *SimSharedState, frame_rate: f32) !void {
    var frame: f32 = 0.0;
    const sim_dt = 1.0 / frame_rate;
    var last = try std.time.Instant.now();

    while (shared.running.load(.acquire)) {
        const now = try std.time.Instant.now();
        const since_f32 = @as(f32, @floatFromInt(now.since(last)));
        const sim_dt_ns = sim_dt * @as(f32, @floatFromInt(std.time.ns_per_s));
        if (since_f32 < sim_dt_ns) {
            std.time.sleep(1_000_000); // 1ms to avoid busy wait
            continue;
        }
        last = now;

        const current_ready = shared.ready_idx.load(.acquire);
        const write_idx: usize = 1 - current_ready;

        const wd = &shared.world_dev;
        rc.launch_raycast(
            try shared.frame_buffer_dev.view(3, .{ shared.cam.image_height, shared.cam.image_width, 3 }),
            &shared.cam,
            try wd.spheres.view(1, .{wd.spheres.len}),
            try wd.vb.pos_buf.view(2, .{ wd.vb.pos_buf.len / 4, 4 }),
            try wd.vb.color_buf.view(2, .{ wd.vb.color_buf.len / 4, 4 }),
            try wd.vb.normal_buf.view(2, .{ wd.vb.normal_buf.len / 4, 4 }),
            try wd.indices.view(1, .{shared.world.mesh_atlas.indices.items.len}),
            try wd.meshes.view(1, .{shared.world.mesh_atlas.meshes.items.len}),
            try wd.materials.view(1, .{shared.world.materials.items.len}),
        );
        try shared.frame_buffer_dev.toHost(shared.frame_buffers_host[write_idx]);

        shared.ready_idx.store(write_idx, .release);

        frame += 1.0;
    }
}
