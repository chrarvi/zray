pub const Vec3 = [3]f32;
pub const Mat4 = [4][4]f32;

pub const PI: f32 = 3.14159265358979323846264338327950288;

pub fn mat4_zeros() Mat4 {
    return .{
        .{ 0.0, 0.0, 0.0, 0.0 },
        .{ 0.0, 0.0, 0.0, 0.0 },
        .{ 0.0, 0.0, 0.0, 0.0 },
        .{ 0.0, 0.0, 0.0, 0.0 },
    };
}
pub fn mat4_ident() Mat4 {
    return .{
        .{ 1.0, 0.0, 0.0, 0.0 },
        .{ 0.0, 1.0, 0.0, 0.0 },
        .{ 0.0, 0.0, 1.0, 0.0 },
        .{ 0.0, 0.0, 0.0, 1.0 },
    };
}

pub fn cross(a: Vec3, b: Vec3) Vec3 {
    return .{
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    };
}

pub fn dot(a: Vec3, b: Vec3) f32 {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

pub fn length(v: Vec3) f32 {
    return @sqrt(dot(v, v));
}

pub fn normalize(v: Vec3) Vec3 {
    const rlen = 1.0 / length(v);
    return .{ v[0] * rlen, v[1] * rlen, v[2] * rlen };
}

pub fn scale(v: Vec3, c: f32) Vec3 {
    return .{v[0] * c, v[1] * c, v[2] * c};
}

pub fn sub(a: Vec3, b: Vec3) Vec3 {
    return .{ a[0] - b[0], a[1] - b[1], a[2] - b[2] };
}

pub fn add(a: Vec3, b: Vec3) Vec3 {
    return .{ a[0] + b[0], a[1] + b[1], a[2] + b[2] };
}

pub fn mat4_projection_perspective(fovy: f32, aspect: f32, z_near: f32, z_far: f32) Mat4 {
    var out = mat4_zeros();
    const f = 1.0 / @tan(fovy / 2.0);
    const z_range = z_near - z_far;

    out[0][0] = f / aspect;
    out[1][1] = f;
    out[2][2] = (z_far + z_near) / z_range;
    out[2][3] = -1;          // the -1 in the last column (before transpose) moves here
    out[3][2] = (2*z_far*z_near)/z_range;

    return out;
}

pub fn mat4_projection_perspective_inverse(fovy: f32, aspect: f32, z_near: f32, z_far: f32) Mat4 {
    var out = mat4_zeros();
    const t = @tan(fovy / 2.0);
    const z_range = z_near - z_far;

    out[0][0] = aspect * t;
    out[1][1] = t;

    out[2][3] = z_range / (2.0 * z_far * z_near);
    out[3][2] = -1.0;
    out[3][3] = (z_far + z_near) / (2.0 * z_far * z_near);

    return out;
}


pub fn deg2Rad(degrees: f32) f32 {
    return (degrees / 360.0) * 2 * PI;
}
