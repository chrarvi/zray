const std = @import("std");
const stbiw = @import("stb_image_write");

extern fn launch_raycast(img: [*]u8, width: c_int, height: c_int) ?void;

const vec3f = @Vector(3, f32);

const HitRecord = struct {
    point: vec3f,
    normal: vec3f,
    t: f32,
    front_face: bool,

    pub fn set_face_normal(self: *HitRecord, ray: *const Ray, outward_normal: vec3f) void {
        self.front_face = vec3f_dot(ray.dir, outward_normal) < 0.0;
        self.normal = if (self.front_face) outward_normal else -outward_normal;
    }
};

const Sphere = struct {
    center: vec3f,
    radius: f32,

    fn hit(self: *const Sphere, ray: *const Ray) ?HitRecord {
        const oc = self.center - ray.origin;
        const a = vec3f_dot(ray.dir, ray.dir);
        const h = vec3f_dot(ray.dir, oc);
        const c = vec3f_dot(oc, oc) - self.radius * self.radius;
        const disc = h * h - a * c;

        if (disc < 0.0) {
            return null;
        }

        const sqrtd = @sqrt(disc);
        var root = (h - sqrtd) / a;
        if ((root <= RAY_TMIN) or (RAY_TMAX <= root)) {
            root = (h + sqrtd) / a;
            if ((root < RAY_TMIN) or (RAY_TMAX <= root)) {
                return null;
            }
        }

        const point = ray.at(root);
        var record = HitRecord {
            .t = root,
            .point = point,
            .normal = .{0.0, 0.0, 0.0},
            .front_face = false,
        };
        const outward_normal = (point - self.center) / @as(vec3f, @splat(self.radius));
        record.set_face_normal(ray, outward_normal);

        return record;
    }
};

const RAY_TMIN = 0.0;
const RAY_TMAX = 1.0;

const Ray = struct {
    origin: vec3f,
    dir: vec3f,

    fn at(self: *const Ray, t: f32) vec3f {
        return self.origin + @as(vec3f, @splat(t)) * self.dir;
    }
};

fn vec3f_norm(vec: vec3f) f32 {
    return @sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
}

fn vec3f_normalize(vec: vec3f) vec3f {
    const norm = vec3f_norm(vec);
    return vec / @as(vec3f, @splat(norm));
}

fn vec3f_dot(a: vec3f, b: vec3f) f32 {
    const c = a * b;
    return c[0] + c[1] + c[2];
}

fn ray_color(ray: *const Ray, sphere: *const Sphere) vec3f {
    const hit = sphere.hit(ray);

    if (hit != null) {
        if (hit.?.t > 0.0) {
            const hit_normal = vec3f_normalize(ray.at(hit.?.t) - vec3f{ 0.0, 0.0, -1.0 });
            return @as(vec3f, @splat(0.5)) * (hit_normal + @as(vec3f, @splat(1.0)));
        }
    }
    const unit_dir = vec3f_normalize(ray.dir);
    const a = 0.5 * (unit_dir[1] + 1.0);
    const a_v = @as(vec3f, @splat(a));
    return @as(vec3f, @splat(1.0 - a)) * @as(vec3f, @splat(1.0)) + a_v * vec3f{ 0.5, 0.7, 1.0 };
}

pub fn main() !void {
    var allocator = std.heap.page_allocator;

    const aspect_ratio = 16.0 / 9.0;
    const image_width = 400;
    const image_height: i32 = @max(@divFloor(image_width, aspect_ratio), 1);
    const img = try allocator.alloc(u8, image_width * image_height * 3);
    defer allocator.free(img);

    const viewport_height: f32 = 2.0;
    const viewport_width = viewport_height * @as(f32, @floatFromInt(image_width)) / @as(f32, @floatFromInt(image_height));

    const focal_length = 1.0;
    const camera_center = vec3f{ 0.0, 0.0, 0.0 };

    const viewport_u = vec3f{ viewport_width, 0.0, 0.0 };
    const viewport_v = vec3f{ 0.0, -viewport_height, 0.0 };

    const pixel_delta_u = viewport_u / @as(vec3f, @splat(@as(f32, @floatFromInt(image_width))));
    const pixel_delta_v = viewport_v / @as(vec3f, @splat(@as(f32, @floatFromInt(image_height))));

    const two = @as(vec3f, @splat(2.0));
    const viewport_upper_left = camera_center - vec3f{ 0.0, 0.0, focal_length } - viewport_u / two - viewport_v / two;
    const pixel00_loc = viewport_upper_left + (pixel_delta_u + pixel_delta_v) / two;

    const sphere = Sphere{
        .center = vec3f{ 0.0, 0.0, -1.0 },
        .radius = 0.5,
    };
    for (0..image_height) |img_y| {
        for (0..image_width) |img_x| {
            const x_vf32 = @as(vec3f, @splat(@as(f32, @floatFromInt(img_x))));
            const y_vf32 = @as(vec3f, @splat(@as(f32, @floatFromInt(img_y))));
            const pixel_center = pixel00_loc + (x_vf32 * pixel_delta_u) + (y_vf32 * pixel_delta_v);
            const ray_direction = pixel_center - camera_center;
            const ray = Ray{ .dir = ray_direction, .origin = camera_center };
            const color = ray_color(&ray, &sphere);
            const idx = 3 * (image_width * img_y + img_x);

            img[idx + 0] = @intFromFloat(255.0 * @max(@min(1.0, color[0]), 0.0));
            img[idx + 1] = @intFromFloat(255.0 * @max(@min(1.0, color[1]), 0.0));
            img[idx + 2] = @intFromFloat(255.0 * @max(@min(1.0, color[2]), 0.0));
        }
    }

    if (stbiw.stbi_write_png("output.png", image_width, image_height, 3, img.ptr, image_width * 3) == 0) {
        return error.ImageWriteFailed;
    }
}
