{ pkgs, crane, fenix, system, libclang, stdenv, lib, cmake, ... }:
let
  toolchain = with fenix.packages.${system};
    combine [
      minimal.rustc
      minimal.cargo
      targets.x86_64-pc-windows-gnu.latest.rust-std
    ];

  craneLib = (crane.mkLib pkgs).overrideToolchain toolchain;

  my-crate = craneLib.buildPackage {
    src = craneLib.cleanCargoSource ./.;

    nativeBuildInputs = [ cmake ];

    strictDeps = true;
    doCheck = false;

    CARGO_BUILD_TARGET = "x86_64-pc-windows-gnu";

    # fixes issues related to libring
    TARGET_CC = "${pkgs.pkgsCross.mingwW64.stdenv.cc}/bin/${pkgs.pkgsCross.mingwW64.stdenv.cc.targetPrefix}cc";

    #fixes issues related to openssl
    OPENSSL_DIR = "${pkgs.openssl.dev}";
    OPENSSL_LIB_DIR = "${pkgs.openssl.out}/lib";
    OPENSSL_INCLUDE_DIR = "${pkgs.openssl.dev}/include/";

    depsBuildBuild = with pkgs; [
      pkgsCross.mingwW64.stdenv.cc
      pkgsCross.mingwW64.windows.pthreads
    ];

    env.LIBCLANG_PATH = "${libclang.lib}/lib/libclang.so";
    # See: https://hoverbear.org/blog/rust-bindgen-in-nix/
    preBuild = ''
      # From: https://github.com/NixOS/nixpkgs/blob/1fab95f5190d087e66a3502481e34e15d62090aa/pkgs/applications/networking/browsers/firefox/common.nix#L247-L253
      # Set C flags for Rust's bindgen program. Unlike ordinary C
      # compilation, bindgen does not invoke $CC directly. Instead it
      # uses LLVM's libclang. To make sure all necessary flags are
      # included we need to look in a few places.
      export BINDGEN_EXTRA_CLANG_ARGS="$(< ${stdenv.cc}/nix-support/libc-crt1-cflags) \
        $(< ${stdenv.cc}/nix-support/libc-cflags) \
        $(< ${stdenv.cc}/nix-support/cc-cflags) \
        $(< ${stdenv.cc}/nix-support/libcxx-cxxflags) \
        ${lib.optionalString stdenv.cc.isClang "-idirafter ${stdenv.cc.cc}/lib/clang/${lib.getVersion stdenv.cc.cc}/include"} \
        ${lib.optionalString stdenv.cc.isGNU "-isystem ${stdenv.cc.cc}/include/c++/${lib.getVersion stdenv.cc.cc} -isystem ${stdenv.cc.cc}/include/c++/${lib.getVersion stdenv.cc.cc}/${stdenv.hostPlatform.config} -idirafter ${stdenv.cc.cc}/lib/gcc/${stdenv.hostPlatform.config}/${lib.getVersion stdenv.cc.cc}/include"} \
      "
    '';
  };
in
my-crate
