pub const Mat4 = [4][4]f32;

pub const Vec4 = extern struct {
    x: f32, y: f32, z: f32, w: f32,

    pub fn new(x: f32, y: f32, z: f32, w: f32) Vec4 {
        return Vec4{ .x = x, .y = y, .z = z, .w = w };
    }

    pub fn xyz(s: Vec4) Vec3 {
        return Vec3.new(s.x, s.y, s.z);
    }
};

pub const Vec3 = extern struct {
    x: f32,
    y: f32,
    z: f32,

    pub fn new(x: f32, y: f32, z: f32) Vec3 {
        return Vec3{ .x = x, .y = y, .z = z };
    }
    pub fn full(v: f32) Vec3 {
        return Vec3.new(v, v, v);
    }
    pub fn get(self: *const Vec3, idx: usize) f32 {
        return switch (idx) {
            0 => self.x,
            1 => self.y,
            2 => self.z,
            else => unreachable,
        };
    }
    pub fn dot(s: Vec3, o: Vec3) f32 {
        return s.x * o.x + s.y * o.y + s.z * o.z;
    }
    pub fn cross(a: Vec3, b: Vec3) Vec3 {
        return Vec3.new(
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x,
        );
    }
    pub fn length(self: Vec3) f32 {
        return @sqrt(self.dot(self));
    }
    pub fn normalize(self: *Vec3) void {
        const rlen = 1.0 / self.length();
        self.x *= rlen;
        self.y *= rlen;
        self.z *= rlen;
    }

    pub fn scale(s: Vec3, c: f32) Vec3 {
        return Vec3.new(s.x * c, s.y * c, s.z * c);
    }

    pub fn sub(s: Vec3, o: Vec3) Vec3 {
        return Vec3.new(s.x - o.x, s.y - o.y, s.z - o.z);
    }

    pub fn add(s: Vec3, o: Vec3) Vec3 {
        return Vec3.new(s.x + o.x, s.y + o.y, s.z + o.z);
    }
    pub fn divc(s: Vec3, c: f32) Vec3 {
        return s.mulc(1/c);
    }
    pub fn mulc(s: Vec3, c: f32) Vec3 {
        return Vec3.new(s.x * c, s.y * c, s.z * c);
    }
    pub fn min(s: Vec3, o: Vec3) Vec3 {
        return Vec3.new(
            @min(s.x, o.x),
            @min(s.y, o.y),
            @min(s.z, o.z),
        );
    }
    pub fn max(s: Vec3, o: Vec3) Vec3 {
        return Vec3.new(
            @max(s.x, o.x),
            @max(s.y, o.y),
            @max(s.z, o.z),
        );
    }

    pub fn to_vec4(s: Vec3, w: f32) Vec4 {
        return Vec4.new(s.x, s.y, s.z, w);
    }
};

pub const PI: f32 = 3.14159265358979323846264338327950288;

pub fn deg2Rad(degrees: f32) f32 {
    return (degrees / 360.0) * 2 * PI;
}

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

pub fn mat4_projection_perspective(fovy: f32, aspect: f32, z_near: f32, z_far: f32) Mat4 {
    var out = mat4_zeros();
    const f = 1.0 / @tan(fovy / 2.0);
    const z_range = z_near - z_far;

    out[0][0] = f / aspect;
    out[1][1] = f;
    out[2][2] = (z_far + z_near) / z_range;
    out[2][3] = -1; // the -1 in the last column (before transpose) moves here
    out[3][2] = (2 * z_far * z_near) / z_range;

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

pub fn mat4_translate(M: *Mat4, t: Vec3) *Mat4 {
    M[0][3] += t.x;
    M[1][3] += t.y;
    M[2][3] += t.z;
    return M;
}

pub fn mat4_scale(M: *Mat4, s: Vec3) void {
    M[0][0] *= s.x;
    M[1][1] *= s.y;
    M[2][2] *= s.z;
}

pub fn mat4_inverse(M: Mat4) ?Mat4 {
    var inv: [16]f32 = undefined;
    var m: [16]f32 = undefined;

    // Flatten 4x4 matrix into 1D array (row-major)
    for (0..4) |row| {
        for (0..4) |col| {
            m[row*4 + col] = M[row][col];
        }
    }

    inv[0] = m[5]*m[10]*m[15] - m[5]*m[11]*m[14] - m[9]*m[6]*m[15] +
             m[9]*m[7]*m[14] + m[13]*m[6]*m[11] - m[13]*m[7]*m[10];

    inv[4] = -m[4]*m[10]*m[15] + m[4]*m[11]*m[14] +
             m[8]*m[6]*m[15] - m[8]*m[7]*m[14] - m[12]*m[6]*m[11] +
             m[12]*m[7]*m[10];

    inv[8] = m[4]*m[9]*m[15] - m[4]*m[11]*m[13] - m[8]*m[5]*m[15] +
             m[8]*m[7]*m[13] + m[12]*m[5]*m[11] - m[12]*m[7]*m[9];

    inv[12] = -m[4]*m[9]*m[14] + m[4]*m[10]*m[13] +
               m[8]*m[5]*m[14] - m[8]*m[6]*m[13] - m[12]*m[5]*m[10] +
               m[12]*m[6]*m[9];

    inv[1] = -m[1]*m[10]*m[15] + m[1]*m[11]*m[14] +
             m[9]*m[2]*m[15] - m[9]*m[3]*m[14] - m[13]*m[2]*m[11] +
             m[13]*m[3]*m[10];

    inv[5] = m[0]*m[10]*m[15] - m[0]*m[11]*m[14] - m[8]*m[2]*m[15] +
             m[8]*m[3]*m[14] + m[12]*m[2]*m[11] - m[12]*m[3]*m[10];

    inv[9] = -m[0]*m[9]*m[15] + m[0]*m[11]*m[13] + m[8]*m[1]*m[15] -
             m[8]*m[3]*m[13] - m[12]*m[1]*m[11] + m[12]*m[3]*m[9];

    inv[13] = m[0]*m[9]*m[14] - m[0]*m[10]*m[13] - m[8]*m[1]*m[14] +
              m[8]*m[2]*m[13] + m[12]*m[1]*m[10] - m[12]*m[2]*m[9];

    inv[2] = m[1]*m[6]*m[15] - m[1]*m[7]*m[14] - m[5]*m[2]*m[15] +
             m[5]*m[3]*m[14] + m[13]*m[2]*m[7] - m[13]*m[3]*m[6];

    inv[6] = -m[0]*m[6]*m[15] + m[0]*m[7]*m[14] + m[4]*m[2]*m[15] -
             m[4]*m[3]*m[14] - m[12]*m[2]*m[7] + m[12]*m[3]*m[6];

    inv[10] = m[0]*m[5]*m[15] - m[0]*m[7]*m[13] - m[4]*m[1]*m[15] +
              m[4]*m[3]*m[13] + m[12]*m[1]*m[7] - m[12]*m[3]*m[5];

    inv[14] = -m[0]*m[5]*m[14] + m[0]*m[6]*m[13] + m[4]*m[1]*m[14] -
              m[4]*m[2]*m[13] - m[12]*m[1]*m[6] + m[12]*m[2]*m[5];

    inv[3] = -m[1]*m[6]*m[11] + m[1]*m[7]*m[10] + m[5]*m[2]*m[11] -
             m[5]*m[3]*m[10] - m[9]*m[2]*m[7] + m[9]*m[3]*m[6];

    inv[7] = m[0]*m[6]*m[11] - m[0]*m[7]*m[10] - m[4]*m[2]*m[11] +
             m[4]*m[3]*m[10] + m[8]*m[2]*m[7] - m[8]*m[3]*m[6];

    inv[11] = -m[0]*m[5]*m[11] + m[0]*m[7]*m[9] + m[4]*m[1]*m[11] -
              m[4]*m[3]*m[9] - m[8]*m[1]*m[7] + m[8]*m[3]*m[5];

    inv[15] = m[0]*m[5]*m[10] - m[0]*m[6]*m[9] - m[4]*m[1]*m[10] +
              m[4]*m[2]*m[9] + m[8]*m[1]*m[6] - m[8]*m[2]*m[5];

    const det = m[0]*inv[0] + m[1]*inv[4] + m[2]*inv[8] + m[3]*inv[12];
    if (det == 0.0) return null;

    const inv_det = 1.0 / det;

    var result: Mat4 = undefined;
    for (0..16) |i| result[i / 4][i % 4] = inv[i] * inv_det;

    return result;
}



pub fn transform_pos(m: Mat4, v: Vec3) Vec3 {
    return Vec3.new(
        m[0][0]*v.x + m[0][1]*v.y + m[0][2]*v.z + m[0][3],
        m[1][0]*v.x + m[1][1]*v.y + m[1][2]*v.z + m[1][3],
        m[2][0]*v.x + m[2][1]*v.y + m[2][2]*v.z + m[2][3],
    );
}

pub fn transform_normal(m: Mat4, n: Vec3) Vec3 {
    // Upper-left 3x3
    return Vec3{
        .x = m[0][0]*n.x + m[0][1]*n.y + m[0][2]*n.z,
        .y = m[1][0]*n.x + m[1][1]*n.y + m[1][2]*n.z,
        .z = m[2][0]*n.x + m[2][1]*n.y + m[2][2]*n.z,
    };
}
