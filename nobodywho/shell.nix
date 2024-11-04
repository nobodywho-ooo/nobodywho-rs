{ pkgs ? import <nixpkgs> {}, ... }: 
pkgs.mkShell {
  env.LIBCLANG_PATH = "${pkgs.libclang.lib}/lib/libclang.so";
  packages = [
    pkgs.vulkan-headers
    pkgs.cmake
    pkgs.clang
    pkgs.rustup
  ];
}
