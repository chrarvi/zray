/// BVH implementation.
/// Translated from https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/
const al = @import("linalg.zig");
const core = @import("core.zig");
const std = @import("std");

const TEST_MESH_IDX: usize = 0;

pub const BoundingVolumeHierarchy = struct {
    const Self = @This();
    const NODE_CAPACITY = 128;
    const PRIM_CAPACITY_PER_NODE = 64;
    const TRIS_CAPACITY = 256;

    const AABB = struct {
        min: al.Vec3,
        max: al.Vec3,

        pub fn unbounded() AABB {
            const fmin = std.math.floatMin(f32);
            const fmax = std.math.floatMax(f32);
            return AABB{
                .min = .{ fmax, fmax, fmax },
                .max = .{ fmin, fmin, fmin },
            };
        }
    };
    const Node = struct {
        box: AABB,
        left_idx: ?usize,
        right_idx: ?usize,
        prims_offset: usize,
        prims_count: usize,

        fn is_leaf(self: *const Node) bool {
            return self.prims_count > 0;
        }
    };

    nodes: [NODE_CAPACITY]Node,
    root_idx: usize,
    node_count: usize,

    prim_indices: [TRIS_CAPACITY]usize,

    pub fn init() Self {
        return Self{
            .nodes = undefined,
            .node_count = 0,
            .root_idx = 0,
            .prim_indices = undefined,
        };
    }

    pub fn build(self: *Self, atlas: *const core.MeshAtlas) void {
        const n_tris = atlas.num_triangles(TEST_MESH_IDX);
        for (0..n_tris) |ti| {
            // compute centroid
            self.prim_indices[ti] = ti;
        }
        self.node_count = 1;
        var root = self.get_node_mut(self.root_idx);
        root.box = AABB.unbounded();
        root.left_idx = null;
        root.right_idx = null;
        root.prims_offset = 0;
        root.prims_count = n_tris;

        self.update_node_aabb(atlas, self.root_idx);
        self.subdivide(atlas, self.root_idx);
    }

    fn get_node_mut(self: *Self, node_idx: usize) *Node {
        std.debug.assert((node_idx < self.node_count) and (node_idx < NODE_CAPACITY));
        return &self.nodes[node_idx];
    }

    fn update_node_aabb(self: *Self, atlas: *const core.MeshAtlas, node_idx: usize) void {
        var node = self.get_node_mut(node_idx);
        node.box = AABB.unbounded();

        for (0..node.prims_count) |pi| {
            const leaf_tri_idx = self.prim_indices[node.prims_offset + pi];
            const leaf_tri = &atlas.get_triangle(TEST_MESH_IDX, leaf_tri_idx).?;
            node.box.min = al.vec3_min(node.box.min, leaf_tri.pos[0]);
            node.box.min = al.vec3_min(node.box.min, leaf_tri.pos[1]);
            node.box.min = al.vec3_min(node.box.min, leaf_tri.pos[2]);

            node.box.max = al.vec3_max(node.box.max, leaf_tri.pos[0]);
            node.box.max = al.vec3_max(node.box.max, leaf_tri.pos[1]);
            node.box.max = al.vec3_max(node.box.max, leaf_tri.pos[2]);
        }
    }

    fn subdivide(self: *Self, atlas: *const core.MeshAtlas, node_idx: usize) void {
        var node = self.get_node_mut(node_idx);

        if (node.prims_count <= 2) {
            return;
        }

        const extent = al.sub(node.box.max, node.box.min);
        var axis: usize = 0;
        if (extent[1] > extent[0]) axis = 1;
        if (extent[2] > extent[axis]) axis = 2;
        const split_pos = node.box.min[axis] + extent[axis] * 0.5;

        // partition into two groups (quicksort ish)
        var i = node.prims_offset;
        var j = i + node.prims_count - 1;
        while (i <= j) {
            const tri = atlas.get_triangle(TEST_MESH_IDX, self.prim_indices[i]).?;
            const centroid = al.scale(al.add(al.add(tri.pos[0], tri.pos[1]), tri.pos[2]), 1.0 / 3.0);
            if (centroid[axis] < split_pos) {
                i += 1;
            } else {
                const tmp = self.prim_indices[i];
                self.prim_indices[i] = self.prim_indices[j];
                self.prim_indices[j] = tmp;
                j -= 1;
            }
        }

        const left_count: usize = i - node.prims_offset;
        if ((left_count == 0) or (left_count == node.prims_count)) return;

        const left_child_idx = self.node_count;
        self.node_count += 1;
        const right_child_idx = self.node_count;
        self.node_count += 1;

        node.left_idx = left_child_idx;
        var left_node = self.get_node_mut(left_child_idx);
        var right_node = self.get_node_mut(right_child_idx);
        left_node.prims_offset = node.prims_offset;
        left_node.prims_count = left_count;
        right_node.prims_offset = i;
        right_node.prims_count = node.prims_count - left_count;
        node.prims_count = 0; // turns it into an internal node

        self.update_node_aabb(atlas, left_child_idx);
        self.update_node_aabb(atlas, right_child_idx);
        self.subdivide(atlas, left_child_idx);
        self.subdivide(atlas, right_child_idx);
    }

    pub fn print(self: *const Self) void {
        std.debug.print("BVH: {?}", self);

    }
};
