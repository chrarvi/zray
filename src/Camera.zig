const al = @import("math.zig");

const Self = @This();

eye: al.Vec3,
target: al.Vec3,
up: al.Vec3,

pub fn init(eye: al.Vec3, target: al.Vec3, up: al.Vec3) Self {
    return .{ .eye = eye, .target = target, .up = up };
}

pub fn look_at(self: *const Self) al.Mat4 {
    const f = al.normalize(self.target - self.eye);
    const r = al.normalize(al.cross(f, self.up));
    const u = al.cross(r, f);

    return al.Mat4{
        .{ r[0], u[0], -f[0], 0.0 },
        .{ r[1], u[1], -f[1], 0.0 },
        .{ r[2], u[2], -f[2], 0.0 },
        .{ -al.dot(r, self.eye), -al.dot(u, self.eye), al.dot(f, self.eye), 1.0 },
    };
}

pub fn camera_to_world(self: *const Self) al.Mat4 {
    const f = al.normalize(al.sub(self.target, self.eye));
    const r = al.normalize(al.cross(f, self.up));
    const u = al.cross(r, f);

    return al.Mat4{
        .{ r[0], r[1], r[2], 0.0 },
        .{ u[0], u[1], u[2], 0.0 },
        .{ -f[0], -f[1], -f[2], 0.0 },
        .{ self.eye[0], self.eye[1], self.eye[2], 1.0 },
    };
}
