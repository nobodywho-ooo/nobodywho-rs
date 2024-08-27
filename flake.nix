{
  description = "A very basic flake";
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
    let
      pkgs = (import nixpkgs { system = system; });
    in
  {
      packages.default = pkgs.callPackage ./nobody {};
      packages.windows = pkgs.pkgsCross.mingwW64.callPackage ./nobody {};
  });
}
