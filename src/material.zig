pub const MaterialKind = enum(u32) {
    LAMBERTIAN = 0,
    METAL = 1,
};

pub const Material = extern struct {
    kind: MaterialKind,
    albedo: [3]f32,
    fuzz: f32 = 0.0,
};
