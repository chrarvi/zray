const cu = @import("cuda.zig");

const TensorView = cu.TensorView;

pub extern fn add_f32_2d(a: TensorView(f32, 2), b: TensorView(f32, 2), b: TensorView(f32, 2)) void;
pub extern fn add_i32_2d(a: TensorView(i32, 2), b: TensorView(i32, 2), b: TensorView(i32, 2)) void;
