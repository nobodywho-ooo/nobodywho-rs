{ pkgs ? import <nixpkgs> { }, ... }: 
let
  android_ndk = "${pkgs.androidenv.androidPkgs.ndk-bundle}/libexec/android-sdk/ndk/27.0.12077973";
in
pkgs.mkShell {
  env.LIBCLANG_PATH = "${pkgs.libclang.lib}/lib/libclang.so";
  env.ANDROID_NDK = android_ndk;

  # https://godot-rust.github.io/book/toolchain/export-android.html
  env.CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER = "${android_ndk}/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang";
  env.CLANG_PATH = "${android_ndk}/toolchains/llvm/prebuilt/linux-x86_64/bin/clang";

  packages = [
    pkgs.cmake
    pkgs.clang
    pkgs.rustup

    # these are the dependencies required by llama.cpp to build for vulkan
    # (these packages were discovered by looking at the nix source code in ggerganov/llama.cpp)
    pkgs.vulkan-headers
    pkgs.vulkan-loader
    pkgs.shaderc

    # for llama-cpp-rs on aarch64-linux-android
    pkgs.glibc_multi.dev
    pkgs.androidenv.androidPkgs.ndk-bundle
  ];
  shellHook = ''
    ulimit -n 2048
  '';
}
