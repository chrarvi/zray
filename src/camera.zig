const std = @import("std");
const al = @import("math.zig");

pub const Camera = struct {
    fov: f32,
    aspect: f32,
    z_near: f32,
    z_far: f32,

    pos: al.Vec3 = .{ 0, 0, 10 },
    front: al.Vec3 = .{ 0, 0, -1 },
    up: al.Vec3 = .{ 0, 1, 0 },
    right: al.Vec3 = .{ 1, 0, 0 },

    yaw: f32 = 0.0,
    pitch: f32 = 0.0,
    speed: f32 = 0.2,

    mouse_position: al.Vec3,
    mouse_sensitivity: f32 = 0.1,

    can_move: bool = false,

    proj: al.Mat4,
    inv_proj: al.Mat4,

    pub const Dir = enum {
        Up,
        Down,
        Left,
        Right,
        Forward,
        Back,
    };

    pub fn init_default(screen_width: usize, screen_height: usize) Camera {
        const width: f32 = @floatFromInt(screen_width);
        const height: f32 = @floatFromInt(screen_height);
        const aspect = width / height;
        const camera = Camera{
            .z_near = 0.1,
            .z_far = 500.0,
            .fov = 90.0,
            .mouse_position = .{ @as(f32, @floatFromInt(screen_width)) / 2, @as(f32, @floatFromInt(screen_height)) / 2, 0 },
            .aspect = aspect,
            .proj = al.mat4_projection_perspective(90.0, aspect, 0.1, 500.0),
            .inv_proj = al.mat4_projection_perspective_inverse(90.0, aspect, 0.1, 500.0),
        };

        return camera;
    }

    pub fn update(self: *Camera) void {
        const yaw = al.deg2Rad(self.yaw);
        const pitch = al.deg2Rad(self.pitch);

        const cos_yaw = std.math.cos(yaw);
        const sin_yaw = std.math.sin(yaw);
        const cos_pitch = std.math.cos(pitch);
        const sin_pitch = std.math.sin(pitch);

        self.front = al.normalize(.{
            cos_pitch * sin_yaw,
            sin_pitch,
            -cos_pitch * cos_yaw,
        });

        const world_up = .{ 0.0, 1.0, 0.0 };
        self.right = al.normalize(al.cross(self.front, world_up));
        self.up = al.normalize(al.cross(self.right, self.front));
    }

    pub fn set_mouse_position(self: *Camera, x: f32, y: f32) void {
        self.mouse_position = .{ x, y, 0 };
    }

    pub fn camera_to_world(self: *const Camera) al.Mat4 {
        return al.Mat4{
            .{ self.right[0], self.up[0], -self.front[0], self.pos[0] },
            .{ self.right[1], self.up[1], -self.front[1], self.pos[1] },
            .{ self.right[2], self.up[2], -self.front[2], self.pos[2] },
            .{ 0.0,           0.0,        0.0,            1.0       },
        };
    }

    pub fn move(self: *Camera, dir: Dir) void {
        const v = switch (dir) {
            .Up => al.scale(self.up, self.speed),
            .Down => al.scale(self.up, -self.speed),
            .Left => al.scale(self.right, -self.speed),
            .Right => al.scale(self.right, self.speed),
            .Forward => al.scale(self.front, self.speed),
            .Back => al.scale(self.front, -self.speed),
        };

        self.pos = al.add(self.pos, v);
    }
};
