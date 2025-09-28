/// BVH implementation.
/// Translated from https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/
const al = @import("linalg.zig");
const core = @import("core.zig");
const std = @import("std");

const AABB = struct {
    min: al.Vec3,
    max: al.Vec3,

    pub fn empty() AABB {
        const fmin = -std.math.inf(f32);
        const fmax = std.math.inf(f32);
        return AABB{
            .min = al.Vec3.full(fmax),
            .max = al.Vec3.full(fmin),
        };
    }
    pub fn extend(self: *AABB, p: al.Vec3) void {
        self.min = self.min.min(p);
        self.max = self.max.max(p);
    }
    pub fn merge(self: *AABB, other: *const AABB) void {
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
    }
    pub fn extent(self: *const AABB) al.Vec3 {
        return self.max.sub(self.min);
    }
    pub fn center(self: *const AABB) al.Vec3 {
        return self.max.add(self.min).divc(2.0);
    }
};

const BVHNode = struct {
    box: AABB,
    left_idx: i32 = -1,
    prims_offset: u32,
    prims_count: u32,
};


pub const BLASBuilder = struct {
    const Self = @This();

    nodes: std.ArrayList(BVHNode),
    prim_indices: std.ArrayList(u32),

    pub fn init(alloc: std.mem.Allocator) Self {
        return Self{
            .nodes = std.ArrayList(BVHNode).init(alloc),
            .prim_indices = std.ArrayList(u32).init(alloc),
        };
    }
    pub fn deinit(self: *BLASBuilder) void {
        self.nodes.deinit();
        self.prim_indices.deinit();
    }

    pub fn build(self: *Self, atlas: *const core.MeshAtlas, max_depth: u32) !void {
        const n_prims = atlas.num_triangles();
        for (0..n_prims) |ti| {
            try self.prim_indices.append(@as(u32, @intCast(ti)));
        }

        // Do this up front to save ourselves reallocations
        try self.nodes.ensureTotalCapacity(std.math.pow(u32, 2, max_depth + 1) - 1);

        var root = try self.nodes.addOne();
        root.box = AABB.empty();
        root.prims_offset = 0;
        root.prims_count = @as(u32, @intCast(n_prims));
        root.left_idx = -1;
        self.update_node_aabb(atlas, root);

        try self.subdivide(atlas, root, 1, max_depth);
    }

    fn update_node_aabb(self: *Self, atlas: *const core.MeshAtlas, node: *BVHNode) void {
        node.box = AABB.empty();

        for (0..node.prims_count) |pi| {
            const leaf_tri_idx = self.prim_indices.items[node.prims_offset + pi];
            const leaf_tri = &atlas.get_triangle(leaf_tri_idx).?;
            node.box.extend(leaf_tri.pos[0]);
            node.box.extend(leaf_tri.pos[1]);
            node.box.extend(leaf_tri.pos[2]);
        }
    }

    fn subdivide(self: *Self, atlas: *const core.MeshAtlas, node: *BVHNode, current_depth: u32, max_depth: u32) !void {
        if (node.prims_count <= 2) return;
        if (current_depth > max_depth) return;

        const extent = node.box.extent();
        var axis: u32 = 0;
        if (extent.y > extent.x) axis = 1;
        if (extent.z > extent.get(axis)) axis = 2;
        const split_pos = (node.box.min.get(axis) + node.box.max.get(axis)) * 0.5;

        // partition into two groups (quicksort ish)
        var i = node.prims_offset;
        var j = i + node.prims_count - 1;
        while (i <= j) {
            const tri = atlas.get_triangle(self.prim_indices.items[i]).?;
            const centroid = tri.pos[0].add(tri.pos[1]).add(tri.pos[2]).scale(1.0 / 3.0);
            if (centroid.get(axis) < split_pos) {
                i += 1;
            } else {
                if (j == 0) break;
                const tmp = self.prim_indices.items[i];
                self.prim_indices.items[i] = self.prim_indices.items[j];
                self.prim_indices.items[j] = tmp;
                j -= 1;
            }
        }

        const left_count: u32 = i - node.prims_offset;
        if ((left_count == 0) or (left_count == node.prims_count)) return;

        const left_child_idx = self.nodes.items.len;
        var left_node = try self.nodes.addOne();
        left_node.prims_offset = node.prims_offset;
        left_node.prims_count = left_count;

        var right_node = try self.nodes.addOne();
        right_node.prims_offset = i;
        right_node.prims_count = node.prims_count - left_count;

        node.left_idx = @as(i32, @intCast(left_child_idx));
        node.prims_count = 0; // turns it into an internal node

        self.update_node_aabb(atlas, left_node);
        self.update_node_aabb(atlas, right_node);

        try self.subdivide(atlas, left_node, current_depth + 1, max_depth);
        try self.subdivide(atlas, right_node, current_depth + 1, max_depth);
    }
};

pub const TLASBuilder = struct {
    const Self = @This();

    nodes: std.ArrayList(BVHNode),
    prim_indices: std.ArrayList(u32),

    pub fn init(alloc: std.mem.Allocator) Self {
        return Self{
            .nodes = std.ArrayList(BVHNode).init(alloc),
            .prim_indices = std.ArrayList(u32).init(alloc),
        };
    }
    pub fn deinit(self: *TLASBuilder) void {
        self.nodes.deinit();
        self.prim_indices.deinit();
    }

    pub fn build(self: *Self, atlas: *const core.MeshAtlas, max_depth: u32) !void {
        const n_prims = atlas.meshes.items.len;
        for (0..n_prims) |ti| {
            try self.prim_indices.append(@as(u32, @intCast(ti)));
        }

        // Do this up front to save ourselves reallocations
        try self.nodes.ensureTotalCapacity(std.math.pow(u32, 2, max_depth + 1) - 1);

        var root = try self.nodes.addOne();
        root.box = AABB.empty();
        root.prims_offset = 0;
        root.prims_count = @as(u32, @intCast(n_prims));
        root.left_idx = -1;
        self.update_node_aabb(atlas, root);

        try self.subdivide(atlas, root, 1, max_depth);

        for (self.nodes.items, 0..) |node, i| {
            std.debug.print("{}: {any}\n", .{i, node});
        }
    }

    fn update_node_aabb(self: *Self, atlas: *const core.MeshAtlas, node: *BVHNode) void {
        node.box = AABB.empty();
        for (0..node.prims_count) |pi| {
            const mesh_idx = self.prim_indices.items[node.prims_offset + pi];
            const mesh = &atlas.meshes.items[mesh_idx];

            var mesh_box = AABB.empty();
            for (0..mesh.index_count) |vert_idx| {
                const tri = atlas.get_mesh_triangle(mesh_idx,  vert_idx / 3);
                mesh_box.extend(tri.?.pos[0]);
                mesh_box.extend(tri.?.pos[1]);
                mesh_box.extend(tri.?.pos[2]);
            }
            node.box.merge(&mesh_box);
        }
    }

    fn subdivide(self: *Self, atlas: *const core.MeshAtlas, node: *BVHNode, current_depth: u32, max_depth: u32) !void {
        if (node.prims_count <= 2) return;
        if (current_depth > max_depth) return;

        const extent = node.box.extent();
        var axis: u32 = 0;
        if (extent.y > extent.x) axis = 1;
        if (extent.z > extent.get(axis)) axis = 2;
        const split_pos = (node.box.min.get(axis) + node.box.max.get(axis)) * 0.5;

        // partition into two groups (quicksort ish)
        var i = node.prims_offset;
        var j = i + node.prims_count - 1;
        while (i <= j) {
            const mesh_idx = self.prim_indices.items[i];
            const mesh = &atlas.meshes.items[mesh_idx];
            var mesh_box = AABB.empty();
            for (0..mesh.index_count) |vert_idx| {
                const tri = &atlas.get_mesh_triangle(mesh_idx, vert_idx / 3).?;
                mesh_box.extend(tri.pos[0]);
                mesh_box.extend(tri.pos[1]);
                mesh_box.extend(tri.pos[2]);
            }
            const centroid = mesh_box.center();

            if (centroid.get(axis) < split_pos) {
                i += 1;
            } else {
                if (j == 0) break;
                const tmp = self.prim_indices.items[i];
                self.prim_indices.items[i] = self.prim_indices.items[j];
                self.prim_indices.items[j] = tmp;
                j -= 1;
            }
        }

        const left_count: u32 = i - node.prims_offset;
        if ((left_count == 0) or (left_count == node.prims_count)) return;

        const left_child_idx = self.nodes.items.len;
        var left_node = try self.nodes.addOne();
        left_node.prims_offset = node.prims_offset;
        left_node.prims_count = left_count;

        var right_node = try self.nodes.addOne();
        right_node.prims_offset = i;
        right_node.prims_count = node.prims_count - left_count;

        node.left_idx = @as(i32, @intCast(left_child_idx));
        node.prims_count = 0; // turns it into an internal node

        self.update_node_aabb(atlas, left_node);
        self.update_node_aabb(atlas, right_node);

        try self.subdivide(atlas, left_node, current_depth + 1, max_depth);
        try self.subdivide(atlas, right_node, current_depth + 1, max_depth);
    }
};
