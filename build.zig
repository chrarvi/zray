const std = @import("std");

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

    const exe = b.addExecutable(.{
        .name = "main",
        .root_module = b.createModule(
            .{ .root_source_file = b.path("main.zig"), .target = target, .optimize = optimize },
        ),
    });

    exe.root_module.addImport("stb_image_write", stbiw_mod);
    exe.addIncludePath(b.path("external"));
    exe.linkLibC();

    const cuda_gen_step = b.addSystemCommand(&.{
        "nvcc", "-Xcompiler", "-fPIC", "-c", "-o", "raycast.o", "cuda/raycast.cu",
    });
    exe.step.dependOn(&cuda_gen_step.step);
    exe.addObjectFile(b.path("raycast.o"));
    exe.addLibraryPath(.{
        .cwd_relative = "/opt/cuda/lib64/",
    });
    exe.linkSystemLibrary("cudart");

    b.installArtifact(exe);

    const run_exe = b.addRunArtifact(exe);
    const run_step = b.step("run", "Run the application");

    run_step.dependOn(&run_exe.step);
}
