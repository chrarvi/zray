/// BVH implementation.
/// Translated from https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/
const al = @import("linalg.zig");
const core = @import("core.zig");
const std = @import("std");

pub const BoundingVolumeHierarchy = struct {
    const Self = @This();

    const AABB = struct {
        min: al.Vec3,
        max: al.Vec3,

        pub fn unbounded() AABB {
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

    const Node = struct {
        box: AABB,
        left_idx: ?usize = null,
        right_idx: ?usize = null,  // not required, it's always left + 1
        prims_offset: usize,
        prims_count: usize,
        depth: usize,  // debug, get rid of to save space
    };

    nodes: std.ArrayList(Node),
    prim_indices: std.ArrayList(usize),

    pub fn init(alloc: std.mem.Allocator) Self {
        return Self{
            .nodes = std.ArrayList(Node).init(alloc),
            .prim_indices = std.ArrayList(usize).init(alloc),
        };
    }

    pub fn build(self: *Self, atlas: *const core.MeshAtlas, max_depth: usize) !void {
        const n_tris = atlas.num_triangles();
        for (0..n_tris) |ti| {
            try self.prim_indices.append(ti);
        }

        // Do this up front to save ourselves reallocations
        try self.nodes.ensureTotalCapacity(std.math.pow(usize, 2, max_depth + 1) - 1);

        var root = try self.nodes.addOne();
        root.box = AABB.unbounded();
        root.prims_offset = 0;
        root.prims_count = n_tris;
        root.depth = 0;  // del
        self.update_node_aabb(atlas, root);

        try self.subdivide(atlas, root, 1, max_depth);
    }

    fn update_node_aabb(self: *Self, atlas: *const core.MeshAtlas, node: *Node) void {
        node.box = AABB.unbounded();

        for (0..node.prims_count) |pi| {
            const leaf_tri_idx = self.prim_indices.items[node.prims_offset + pi];
            const leaf_tri = &atlas.get_triangle(leaf_tri_idx).?;
            node.box.extend(leaf_tri.pos[0]);
            node.box.extend(leaf_tri.pos[1]);
            node.box.extend(leaf_tri.pos[2]);
        }
    }

    fn subdivide(self: *Self, atlas: *const core.MeshAtlas, node: *Node, current_depth: usize, max_depth: usize) !void {
        if (node.prims_count <= 2) return;
        if (current_depth > max_depth) return;

        const extent = node.box.extent();
        var axis: usize = 0;
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
                const tmp = self.prim_indices.items[i];
                self.prim_indices.items[i] = self.prim_indices.items[j];
                self.prim_indices.items[j] = tmp;
                j -= 1;
            }
        }

        const left_count: usize = i - node.prims_offset;
        if ((left_count == 0) or (left_count == node.prims_count)) return;

        const left_child_idx = self.nodes.items.len;
        var left_node = try self.nodes.addOne();
        left_node.prims_offset = node.prims_offset;
        left_node.prims_count = left_count;
        left_node.depth = current_depth;  // del

        const right_child_idx = self.nodes.items.len;
        var right_node = try self.nodes.addOne();
        right_node.prims_offset = i;
        right_node.prims_count = node.prims_count - left_count;
        right_node.depth = current_depth;  // del

        node.left_idx = left_child_idx;
        node.right_idx = right_child_idx;
        node.prims_count = 0; // turns it into an internal node

        self.update_node_aabb(atlas, left_node);
        self.update_node_aabb(atlas, right_node);

        try self.subdivide(atlas, left_node, current_depth + 1, max_depth);
        try self.subdivide(atlas, right_node, current_depth + 1, max_depth);
    }
};
