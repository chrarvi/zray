const std = @import("std");

pub fn compile_cuda(b: *std.Build, cuda_file: []const u8, obj_file: []const u8) *std.Build.Step {
    const cuda_gen_step = b.addSystemCommand(&.{
        "nvcc", "-Xcompiler", "-fPIC", "-c", "-o", obj_file, cuda_file,
    });
    return &cuda_gen_step.step;
}

pub fn link_cuda(b: *std.Build, exe: *std.Build.Step.Compile) void{
    exe.addLibraryPath(.{
        .cwd_relative = "/opt/cuda/lib64/",
    });
    exe.addIncludePath(.{ .cwd_relative = "/opt/cuda/include" });
    exe.addIncludePath(b.path("cuda"));
    exe.linkSystemLibrary("cudart");
}


pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const gen_step = b.addWriteFiles();
    const stbiw_translate_c = b.addTranslateC(.{
        .root_source_file = b.path("external/stb_image_write.h"),
        .target = target,
        .optimize = optimize,
    });
    const stbiw_mod = stbiw_translate_c.createModule();
    const stbiw_c_path = gen_step.add("stb_image_write.c", "#define STB_IMAGE_WRITE_IMPLEMENTATION\n#include \"stb_image_write.h\"\n");
    stbiw_mod.addCSourceFile(.{ .file = stbiw_c_path });
    stbiw_mod.addIncludePath(b.path("external"));

    const raylib_translate = b.addTranslateC(.{
        .root_source_file = b.path("external/raylib-5.5_linux_amd64/include/raylib.h"),
        .target = target,
        .optimize = optimize,
    });
    const raylib_mod = raylib_translate.createModule();

    const raycast_o = "build/raycast.o";
    const add_o     = "build/add.o";
    const cuda_steps = [_]*std.Build.Step{
        compile_cuda(b, "cuda/raycast.cu", raycast_o),
        compile_cuda(b, "cuda/add.cu", add_o),
    };

    const exe = b.addExecutable(.{
        .name = "main",
        .root_module = b.createModule(
            .{ .root_source_file = b.path("src/main.zig"), .target = target, .optimize = optimize },
        ),
    });

    exe.root_module.addImport("stb_image_write", stbiw_mod);
    exe.root_module.addImport("raylib", raylib_mod);
    exe.addLibraryPath(b.path("external/raylib-5.5_linux_amd64/lib/"));
    exe.linkSystemLibrary("raylib");
    exe.linkLibC();
    link_cuda(b, exe);

    for (cuda_steps) |s| exe.step.dependOn(s);
    exe.addObjectFile(b.path(raycast_o));
    exe.addObjectFile(b.path(add_o));

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
    test_exe.linkLibC();
    link_cuda(b, test_exe);

    for (cuda_steps) |s| test_exe.step.dependOn(s);
    test_exe.addObjectFile(b.path(raycast_o));
    test_exe.addObjectFile(b.path(add_o));

    const test_step = b.step("test", "Run all tests");
    const run_tests = b.addRunArtifact(test_exe);
    test_step.dependOn(&run_tests.step);
}
