const std = @import("std");
const al = @import("linalg.zig");

const Camera = @This();

fov: f32,
aspect: f32,
z_near: f32,
z_far: f32,

pos: al.Vec3 = al.Vec3.new(0, 0, 10),
front: al.Vec3 = al.Vec3.new(0, 0, -1),
up: al.Vec3 = al.Vec3.new(0, 1, 0),
right: al.Vec3 = al.Vec3.new(1, 0, 0),

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
    const width: f32 = @as(f32, @floatFromInt(screen_width));
    const height: f32 = @as(f32, @floatFromInt(screen_height));
    const aspect = width / height;
    const fov = al.deg2Rad(140.0);
    const camera = Camera{
        .z_near = 0.1,
        .z_far = 500.0,
        .fov = fov,
        .mouse_position = al.Vec3.new(width / 2, height / 2, 0),
        .aspect = aspect,
        .proj = al.mat4_projection_perspective(fov, aspect, 0.1, 500.0),
        .inv_proj = al.mat4_projection_perspective_inverse(fov, aspect, 0.1, 500.0),
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

    self.front = al.Vec3.new(
        cos_pitch * sin_yaw,
        sin_pitch,
        -cos_pitch * cos_yaw,
    );
    self.front.normalize();

    const world_up = al.Vec3.new( 0.0, 1.0, 0.0 );
    self.right = self.front.cross(world_up);
    self.right.normalize();
    self.up = self.right.cross(self.front);
    self.up.normalize();
}

pub fn set_mouse_position(self: *Camera, x: f32, y: f32) void {
    self.mouse_position = .{ x, y, 0 };
}

pub fn camera_to_world(self: *const Camera) al.Mat4 {
    return al.Mat4{
        .{ self.right.x, self.up.x, -self.front.x, self.pos.x },
        .{ self.right.y, self.up.y, -self.front.y, self.pos.y },
        .{ self.right.z, self.up.z, -self.front.z, self.pos.z },
        .{ 0.0, 0.0, 0.0, 1.0 },
    };
}

pub fn move(self: *Camera, dir: Dir) void {
    const v = switch (dir) {
        .Up => self.up.scale(self.speed),
        .Down => self.up.scale(-self.speed),
        .Left => self.right.scale(-self.speed),
        .Right => self.right.scale(self.speed),
        .Forward => self.front.scale(self.speed),
        .Back => self.front.scale(-self.speed),
    };

    self.pos = self.pos.add(v);
}
