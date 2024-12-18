{
  description = "NobodyWho - a godot plugin for NPC dialogue with local LLMs";
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
    let
      pkgs = (import nixpkgs {
        system = system;
        config = {
          android_sdk.accept_license = true;
          allowUnfree = true; # You might need this too for Android SDK
        };
      });
    in
  {
      packages.default = pkgs.callPackage ./nobodywho {};
      devShells.default = import ./nobodywho/shell.nix { inherit pkgs; };
  });
}
