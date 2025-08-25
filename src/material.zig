pub const MaterialKind = enum(u32) {
    LAMBERTIAN = 0,
    METAL = 1,
    EMISSIVE = 2,
};

pub const Material = extern struct {
    kind: MaterialKind,
    albedo: [3]f32 = .{1.0, 1.0, 1.0},
    fuzz: f32 = 0.0,
    emit: [3]f32 = .{0.0, 0.0, 0.0},
};
