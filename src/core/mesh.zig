const std = @import("std");
const core = @import("core.zig");
const al = @import("linalg.zig");
const rc = @import("../gpu/raycast.zig");

pub const MeshAtlas = struct {
    vb: core.HostVertexBuffer,
    indices: std.ArrayList(u32),
    meshes: std.ArrayList(rc.Mesh),

    pub fn init(allocator: std.mem.Allocator) MeshAtlas {
        return .{
            .vb = core.HostVertexBuffer.init(allocator),
            .indices = std.ArrayList(u32).init(allocator),
            .meshes = std.ArrayList(rc.Mesh).init(allocator),
        };
    }

    pub fn deinit(self: *MeshAtlas) void {
        self.vb.deinit();
        self.indices.deinit();
        self.meshes.deinit();
    }

    pub fn parse_mesh_from_file(self: *MeshAtlas, filename: []const u8) !*rc.Mesh {
        const file = try std.fs.cwd().openFile(
            filename,
            .{},
        );
        defer file.close();

        var reader = file.reader();

        var tris_buf = try std.BoundedArray(u8, 32).init(32);
        const num_tris: usize = try std.fmt.parseInt(usize, try reader.readUntilDelimiter(&tris_buf.buffer, '\n'), 10);

        // TODO: use scratch arena here
        var bytes = try std.BoundedArray(u8, 10000 * 1024).init(10000 * 1024);
        _ = try reader.readAll(&bytes.buffer);

        var lines_iter = std.mem.splitScalar(u8, &bytes.buffer, '\n');

        const vertex_start = self.vb.pos_buf.items.len;
        const index_start = self.indices.items.len;

        for (0..num_tris) |t_idx| {
            for (0..3) |v_idx| {
                const pos_line = lines_iter.next().?;
                var pos_iter = std.mem.splitScalar(u8, pos_line, ' ');

                const vert: al.Vec3 = .{
                    try std.fmt.parseFloat(f32, pos_iter.next().?),
                    try std.fmt.parseFloat(f32, pos_iter.next().?),
                    try std.fmt.parseFloat(f32, pos_iter.next().?),
                };
                const norm_line = lines_iter.next().?;
                var norm_iter = std.mem.splitScalar(u8, norm_line, ' ');

                const normal: al.Vec3 = .{
                    try std.fmt.parseFloat(f32, norm_iter.next().?),
                    try std.fmt.parseFloat(f32, norm_iter.next().?),
                    try std.fmt.parseFloat(f32, norm_iter.next().?),
                };

                const col: al.Vec3 = .{ 1.0, 1.0, 1.0 };

                try self.vb.push_vertex(vert, col, normal);
                try self.indices.append(@as(u32, @intCast(vertex_start + t_idx * 3 + v_idx)));
            }
            _ = lines_iter.next();
        }

        const mesh = rc.Mesh{
            .index_start = @as(c_uint, @intCast(index_start)),
            .index_count = @as(c_uint, @intCast(self.indices.items.len - index_start)),
            .model = al.mat4_ident(),
            .material_idx = 0,
        };

        try self.meshes.append(mesh);
        return &self.meshes.items[self.meshes.items.len - 1];

    }
};
