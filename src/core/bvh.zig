/// BVH implementation.
/// Translated from https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/
///
/// Two-level acceleration structure:
///   * BLASBuilder builds one BVH per mesh in that mesh's *model* space. All
///     meshes' nodes are concatenated into a single `nodes` buffer; each mesh's
///     sub-range is described by an entry in `meshinfo` (see rc.BLASMeshInfo).
///   * TLASBuilder builds a single BVH over the mesh instances in *world* space
///     (i.e. after each mesh's model transform has been applied). Its leaves
///     reference mesh indices via `prim_indices`.
const al = @import("linalg.zig");
const core = @import("core.zig");
const rc = @import("../gpu/raycast.zig");
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

pub const BVHNode = struct {
    box: AABB,
    // For an internal node (prims_count == 0) this is the absolute index of the
    // left child in the shared node buffer (right child is left_idx + 1).
    // For a leaf it is unused (-1); leaves use prims_offset/prims_count.
    left_idx: i32 = -1,
    // Absolute offset into the builder's prim_indices buffer.
    prims_offset: u32,
    prims_count: u32,
};

pub const BLASBuilder = struct {
    const Self = @This();

    nodes: std.array_list.Managed(BVHNode),
    // Global triangle indices (index into MeshAtlas via get_triangle), grouped
    // by mesh in the same order as `meshinfo`.
    prim_indices: std.array_list.Managed(u32),
    // One entry per mesh describing its node/prim sub-range.
    meshinfo: std.array_list.Managed(rc.BLASMeshInfo),

    pub fn init(alloc: std.mem.Allocator) Self {
        return Self{
            .nodes = std.array_list.Managed(BVHNode).init(alloc),
            .prim_indices = std.array_list.Managed(u32).init(alloc),
            .meshinfo = std.array_list.Managed(rc.BLASMeshInfo).init(alloc),
        };
    }
    pub fn deinit(self: *BLASBuilder) void {
        self.nodes.deinit();
        self.prim_indices.deinit();
        self.meshinfo.deinit();
    }

    pub fn build(self: *Self, atlas: *const core.MeshAtlas, max_depth: u32) !void {
        for (0..atlas.meshes.items.len) |mesh_idx| {
            try self.build_mesh(atlas, mesh_idx, max_depth);
        }
    }

    fn build_mesh(self: *Self, atlas: *const core.MeshAtlas, mesh_idx: usize, max_depth: u32) !void {
        const node_offset = self.nodes.items.len;
        const prim_offset = self.prim_indices.items.len;

        const mesh = atlas.meshes.items[mesh_idx];
        // index_start is a multiple of 3, so the first global triangle index of
        // this mesh is index_start / 3.
        const tri_base: u32 = @intCast(mesh.index_start / 3);
        const n_tris = atlas.num_mesh_triangles(mesh_idx);

        for (0..n_tris) |t| {
            try self.prim_indices.append(tri_base + @as(u32, @intCast(t)));
        }

        // Reserve enough unused capacity so that node pointers obtained via
        // addOne() stay valid for the duration of this mesh's subdivide. A
        // binary BVH over N primitives has at most 2N-1 nodes.
        try self.nodes.ensureUnusedCapacity(2 * n_tris + 1);

        var root = try self.nodes.addOne();
        root.box = AABB.empty();
        root.prims_offset = @intCast(prim_offset);
        root.prims_count = @intCast(n_tris);
        root.left_idx = -1;
        self.update_node_aabb(atlas, root);

        try self.subdivide(atlas, root, 1, max_depth);

        try self.meshinfo.append(.{
            .node_offset = @intCast(node_offset),
            .node_count = @intCast(self.nodes.items.len - node_offset),
            .prim_offset = @intCast(prim_offset),
            .prim_count = @intCast(n_tris),
        });
    }

    fn update_node_aabb(self: *Self, atlas: *const core.MeshAtlas, node: *BVHNode) void {
        node.box = AABB.empty();

        for (0..node.prims_count) |pi| {
            const leaf_tri_idx = self.prim_indices.items[node.prims_offset + pi];
            // get_triangle returns model-space (untransformed) positions.
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

    nodes: std.array_list.Managed(BVHNode),
    // Mesh (instance) indices.
    prim_indices: std.array_list.Managed(u32),

    pub fn init(alloc: std.mem.Allocator) Self {
        return Self{
            .nodes = std.array_list.Managed(BVHNode).init(alloc),
            .prim_indices = std.array_list.Managed(u32).init(alloc),
        };
    }
    pub fn deinit(self: *TLASBuilder) void {
        self.nodes.deinit();
        self.prim_indices.deinit();
    }

    pub fn build(self: *Self, atlas: *const core.MeshAtlas, max_depth: u32) !void {
        const n_prims = atlas.meshes.items.len;
        for (0..n_prims) |mi| {
            try self.prim_indices.append(@as(u32, @intCast(mi)));
        }

        // A binary BVH over N instances has at most 2N-1 nodes; reserve up front
        // so node pointers stay valid across subdivide.
        try self.nodes.ensureTotalCapacity(2 * n_prims + 1);

        var root = try self.nodes.addOne();
        root.box = AABB.empty();
        root.prims_offset = 0;
        root.prims_count = @as(u32, @intCast(n_prims));
        root.left_idx = -1;
        self.update_node_aabb(atlas, root);

        try self.subdivide(atlas, root, 1, max_depth);
    }

    // World-space AABB of a single mesh instance.
    fn mesh_world_aabb(atlas: *const core.MeshAtlas, mesh_idx: u32) AABB {
        var box = AABB.empty();
        const n_tris = atlas.num_mesh_triangles(mesh_idx);
        for (0..n_tris) |t| {
            // get_mesh_triangle returns world-space (model-transformed) positions.
            const tri = atlas.get_mesh_triangle(mesh_idx, t).?;
            box.extend(tri.pos[0]);
            box.extend(tri.pos[1]);
            box.extend(tri.pos[2]);
        }
        return box;
    }

    fn update_node_aabb(self: *Self, atlas: *const core.MeshAtlas, node: *BVHNode) void {
        node.box = AABB.empty();
        for (0..node.prims_count) |pi| {
            const mesh_idx = self.prim_indices.items[node.prims_offset + pi];
            var mesh_box = mesh_world_aabb(atlas, mesh_idx);
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
            const mesh_box = mesh_world_aabb(atlas, mesh_idx);
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
