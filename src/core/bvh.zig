/// BVH implementation.
/// Translated from https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/
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
    // Surface area; 0 for an empty (never-extended) box.
    pub fn area(self: *const AABB) f32 {
        const e = self.extent();
        if (e.x < 0.0 or e.y < 0.0 or e.z < 0.0) return 0.0;
        return 2.0 * (e.x * e.y + e.y * e.z + e.z * e.x);
    }
};

pub const BVHNode = struct {
    box: AABB,
    left_idx: i32 = -1,
    prims_offset: u32,
    prims_count: u32,
    depth: u32 = 0,
};

const SAH_BINS = 12;

const Split = struct {
    axis: u32,
    pos: f32,
    cost: f32,
};

fn find_best_split(
    prim_indices: []const u32,
    first: u32,
    count: u32,
    boxes: []const AABB,
    centroids: []const al.Vec3,
) ?Split {
    var best: ?Split = null;

    var axis: u32 = 0;
    while (axis < 3) : (axis += 1) {
        // Bounds of the centroids along this axis.
        var cmin: f32 = std.math.inf(f32);
        var cmax: f32 = -std.math.inf(f32);
        var k: u32 = 0;
        while (k < count) : (k += 1) {
            const c = centroids[prim_indices[first + k]].get(axis);
            cmin = @min(cmin, c);
            cmax = @max(cmax, c);
        }
        if (cmax - cmin < 1e-12) continue; // no spread: can't split on this axis

        // Bin the primitives by centroid.
        var bin_box: [SAH_BINS]AABB = undefined;
        var bin_count: [SAH_BINS]u32 = undefined;
        for (0..SAH_BINS) |b| {
            bin_box[b] = AABB.empty();
            bin_count[b] = 0;
        }
        const scale: f32 = @as(f32, SAH_BINS) / (cmax - cmin);
        k = 0;
        while (k < count) : (k += 1) {
            const id = prim_indices[first + k];
            var b: usize = @intFromFloat((centroids[id].get(axis) - cmin) * scale);
            if (b >= SAH_BINS) b = SAH_BINS - 1;
            bin_count[b] += 1;
            bin_box[b].merge(&boxes[id]);
        }

        // Sweep to accumulate left/right area*count for each of the BINS-1 planes.
        var left_area: [SAH_BINS - 1]f32 = undefined;
        var right_area: [SAH_BINS - 1]f32 = undefined;
        var left_count: [SAH_BINS - 1]u32 = undefined;
        var right_count: [SAH_BINS - 1]u32 = undefined;

        var lbox = AABB.empty();
        var lcount: u32 = 0;
        for (0..SAH_BINS - 1) |i| {
            lcount += bin_count[i];
            lbox.merge(&bin_box[i]);
            left_count[i] = lcount;
            left_area[i] = lbox.area();
        }
        var rbox = AABB.empty();
        var rcount: u32 = 0;
        var i: usize = SAH_BINS - 1;
        while (i > 0) : (i -= 1) {
            rcount += bin_count[i];
            rbox.merge(&bin_box[i]);
            right_count[i - 1] = rcount;
            right_area[i - 1] = rbox.area();
        }

        const bin_w = (cmax - cmin) / @as(f32, SAH_BINS);
        for (0..SAH_BINS - 1) |plane| {
            // Skip degenerate planes that leave one side empty.
            if (left_count[plane] == 0 or right_count[plane] == 0) continue;
            const cost = @as(f32, @floatFromInt(left_count[plane])) * left_area[plane] +
                @as(f32, @floatFromInt(right_count[plane])) * right_area[plane];
            if (best == null or cost < best.?.cost) {
                best = .{
                    .axis = axis,
                    .pos = cmin + bin_w * @as(f32, @floatFromInt(plane + 1)),
                    .cost = cost,
                };
            }
        }
    }

    return best;
}

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
        const alloc = self.nodes.allocator;

        // Precompute per-triangle model-space box + centroid once, indexed by
        // global triangle index.
        const n_tris_total = atlas.num_triangles();
        const boxes = try alloc.alloc(AABB, n_tris_total);
        defer alloc.free(boxes);
        const centroids = try alloc.alloc(al.Vec3, n_tris_total);
        defer alloc.free(centroids);
        for (0..n_tris_total) |t| {
            // get_triangle returns model-space (untransformed) positions.
            const tri = atlas.get_triangle(t).?;
            var b = AABB.empty();
            b.extend(tri.pos[0]);
            b.extend(tri.pos[1]);
            b.extend(tri.pos[2]);
            boxes[t] = b;
            centroids[t] = tri.pos[0].add(tri.pos[1]).add(tri.pos[2]).scale(1.0 / 3.0);
        }

        for (0..atlas.meshes.items.len) |mesh_idx| {
            try self.build_mesh(atlas, mesh_idx, max_depth, boxes, centroids);
        }
    }

    fn build_mesh(self: *Self, atlas: *const core.MeshAtlas, mesh_idx: usize, max_depth: u32, boxes: []const AABB, centroids: []const al.Vec3) !void {
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
        root.prims_offset = @intCast(prim_offset);
        root.prims_count = @intCast(n_tris);
        root.left_idx = -1;
        root.depth = 0;
        self.update_node_aabb(root, boxes);

        try self.subdivide(root, 1, max_depth, boxes, centroids);

        try self.meshinfo.append(.{
            .node_offset = @intCast(node_offset),
            .node_count = @intCast(self.nodes.items.len - node_offset),
            .prim_offset = @intCast(prim_offset),
            .prim_count = @intCast(n_tris),
        });
    }

    fn update_node_aabb(self: *Self, node: *BVHNode, boxes: []const AABB) void {
        node.box = AABB.empty();
        for (0..node.prims_count) |pi| {
            const id = self.prim_indices.items[node.prims_offset + pi];
            node.box.merge(&boxes[id]);
        }
    }

    fn subdivide(self: *Self, node: *BVHNode, current_depth: u32, max_depth: u32, boxes: []const AABB, centroids: []const al.Vec3) !void {
        if (node.prims_count <= 2) return;
        if (current_depth > max_depth) return;

        const split = find_best_split(self.prim_indices.items, node.prims_offset, node.prims_count, boxes, centroids) orelse return;

        // Stop if splitting is not cheaper than keeping this node as a leaf.
        const leaf_cost = @as(f32, @floatFromInt(node.prims_count)) * node.box.area();
        if (split.cost >= leaf_cost) return;

        // Partition prim_indices in place by the chosen plane.
        var i = node.prims_offset;
        var j = i + node.prims_count - 1;
        while (i <= j) {
            const id = self.prim_indices.items[i];
            if (centroids[id].get(split.axis) < split.pos) {
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

        const child_depth = node.depth + 1;
        const left_child_idx = self.nodes.items.len;
        var left_node = try self.nodes.addOne();
        left_node.prims_offset = node.prims_offset;
        left_node.prims_count = left_count;
        left_node.depth = child_depth;

        var right_node = try self.nodes.addOne();
        right_node.prims_offset = i;
        right_node.prims_count = node.prims_count - left_count;
        right_node.depth = child_depth;

        node.left_idx = @as(i32, @intCast(left_child_idx));
        node.prims_count = 0; // turns it into an internal node

        self.update_node_aabb(left_node, boxes);
        self.update_node_aabb(right_node, boxes);

        try self.subdivide(left_node, current_depth + 1, max_depth, boxes, centroids);
        try self.subdivide(right_node, current_depth + 1, max_depth, boxes, centroids);
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
        const alloc = self.nodes.allocator;
        const n_prims = atlas.meshes.items.len;

        for (0..n_prims) |mi| {
            try self.prim_indices.append(@as(u32, @intCast(mi)));
        }

        // Precompute per-instance world-space box + centroid once, indexed by
        // mesh index.
        const boxes = try alloc.alloc(AABB, n_prims);
        defer alloc.free(boxes);
        const centroids = try alloc.alloc(al.Vec3, n_prims);
        defer alloc.free(centroids);
        for (0..n_prims) |m| {
            boxes[m] = mesh_world_aabb(atlas, @intCast(m));
            centroids[m] = boxes[m].center();
        }

        // A binary BVH over N instances has at most 2N-1 nodes; reserve up front
        // so node pointers stay valid across subdivide.
        try self.nodes.ensureTotalCapacity(2 * n_prims + 1);

        var root = try self.nodes.addOne();
        root.prims_offset = 0;
        root.prims_count = @as(u32, @intCast(n_prims));
        root.left_idx = -1;
        root.depth = 0;
        self.update_node_aabb(root, boxes);

        try self.subdivide(root, 1, max_depth, boxes, centroids);
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

    fn update_node_aabb(self: *Self, node: *BVHNode, boxes: []const AABB) void {
        node.box = AABB.empty();
        for (0..node.prims_count) |pi| {
            const id = self.prim_indices.items[node.prims_offset + pi];
            node.box.merge(&boxes[id]);
        }
    }

    fn subdivide(self: *Self, node: *BVHNode, current_depth: u32, max_depth: u32, boxes: []const AABB, centroids: []const al.Vec3) !void {
        if (node.prims_count <= 2) return;
        if (current_depth > max_depth) return;

        const split = find_best_split(self.prim_indices.items, node.prims_offset, node.prims_count, boxes, centroids) orelse return;

        const leaf_cost = @as(f32, @floatFromInt(node.prims_count)) * node.box.area();
        if (split.cost >= leaf_cost) return;

        var i = node.prims_offset;
        var j = i + node.prims_count - 1;
        while (i <= j) {
            const id = self.prim_indices.items[i];
            if (centroids[id].get(split.axis) < split.pos) {
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

        const child_depth = node.depth + 1;
        const left_child_idx = self.nodes.items.len;
        var left_node = try self.nodes.addOne();
        left_node.prims_offset = node.prims_offset;
        left_node.prims_count = left_count;
        left_node.depth = child_depth;

        var right_node = try self.nodes.addOne();
        right_node.prims_offset = i;
        right_node.prims_count = node.prims_count - left_count;
        right_node.depth = child_depth;

        node.left_idx = @as(i32, @intCast(left_child_idx));
        node.prims_count = 0; // turns it into an internal node

        self.update_node_aabb(left_node, boxes);
        self.update_node_aabb(right_node, boxes);

        try self.subdivide(left_node, current_depth + 1, max_depth, boxes, centroids);
        try self.subdivide(right_node, current_depth + 1, max_depth, boxes, centroids);
    }
};
