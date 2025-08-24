pub const Vec3 = @Vector(3, f32);
pub const Mat4 = [4][4]f32;

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
