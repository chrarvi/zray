const std = @import("std");

pub fn compile_cuda(b: *std.Build, cuda_file: []const u8, obj_file: []const u8) std.Build.LazyPath {
    var cuda_gen_step = b.addSystemCommand(&.{"nvcc"});
    cuda_gen_step.addArgs(&.{ "-Xcompiler", "-fPIC" });
    cuda_gen_step.addArgs(&.{ "-c", cuda_file });
    cuda_gen_step.addFileInput(b.path(cuda_file));
    cuda_gen_step.addArg("-o");
    return cuda_gen_step.addOutputFileArg(obj_file);
}

pub fn link_cuda(b: *std.Build, exe: *std.Build.Step.Compile) void {
    exe.root_module.addLibraryPath(.{
        .cwd_relative = "/opt/cuda/lib64/",
    });
    exe.root_module.addIncludePath(.{ .cwd_relative = "/opt/cuda/include" });
    exe.root_module.addIncludePath(b.path("cuda"));
    exe.root_module.linkSystemLibrary("cudart", .{});
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const raylib_translate = b.addTranslateC(.{
        .root_source_file = b.path("external/raylib-5.5_linux_amd64/include/raylib.h"),
        .target = target,
        .optimize = optimize,
    });
    const raylib_mod = raylib_translate.createModule();
    raylib_mod.addLibraryPath(b.path("external/raylib-5.5_linux_amd64/lib/"));
    raylib_mod.linkSystemLibrary("raylib", .{});

    const mui_translate = b.addTranslateC(.{
        .root_source_file = b.path("external/microui/src/microui.h"),
        .target = target,
        .optimize = optimize,
    });
    const mui_mod = mui_translate.createModule();
    mui_mod.addCSourceFile(.{
        .file = b.path("external/microui/src/microui.c"),
        .flags = &[_][]const u8 {
            "-std=c99",
            "-fno-sanitize=undefined",
        }
    });

    const exe = b.addExecutable(.{
        .name = "main",
        .root_module = b.createModule(
            .{ .root_source_file = b.path("src/main.zig"), .target = target, .optimize = optimize },
        ),
    });

    exe.root_module.addImport("raylib", raylib_mod);
    exe.root_module.addImport("microui", mui_mod);
    exe.root_module.link_libcpp = true;
    link_cuda(b, exe);

    const raycast_o = compile_cuda(b, "cuda/raycast.cu", "build/raycast.o");
    const add_o = compile_cuda(b, "cuda/add.cu", "build/add.o");
    exe.root_module.addObjectFile(raycast_o);
    exe.root_module.addObjectFile(add_o);

    b.installArtifact(exe);

    const run_exe = b.addRunArtifact(exe);
    const run_step = b.step("run", "Run the application");
    run_step.dependOn(&run_exe.step);

    const test_exe = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/test.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    link_cuda(b, test_exe);
    test_exe.root_module.link_libc = true;
    test_exe.root_module.link_libcpp = true;

    test_exe.root_module.addObjectFile(raycast_o);
    test_exe.root_module.addObjectFile(add_o);

    const test_step = b.step("test", "Run all tests");
    const run_tests = b.addRunArtifact(test_exe);
    test_step.dependOn(&run_tests.step);
}
