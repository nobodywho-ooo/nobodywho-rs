{
  description = "A very basic flake";
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    crane.url = "github:ipetkov/crane";
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };
  outputs = { self, nixpkgs, flake-utils, fenix, crane, ... }:
    flake-utils.lib.eachDefaultSystem (system:
    let
      pkgs = (import nixpkgs { system = system; });
    in
  {
      packages.default = pkgs.callPackage ./nobody {};
      packages.windows = pkgs.pkgsCross.mingwW64.callPackage ./nobody {};
      packages.windows-crane = pkgs.callPackage ./nobody/crane.nix { inherit fenix crane pkgs system; };
  });
}
