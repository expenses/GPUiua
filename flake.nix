{
  description = "Flake utils demo";

  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.crane.url = "github:ipetkov/crane";
  inputs.fenix = {
    url = "github:nix-community/fenix";
    inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = {
    self,
    flake-utils,
    nixpkgs,
    crane,
    fenix,
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = nixpkgs.legacyPackages.${system};
        wasm-crane-lib = (crane.mkLib pkgs).overrideToolchain (with fenix.packages.${system};
          combine [
            stable.rustc
            stable.cargo
            targets.wasm32-unknown-unknown.stable.rust-std
          ]);
      in {
        packages = rec {
          wasm = wasm-crane-lib.buildPackage {
            name = "gpuiua";
            src = pkgs.lib.cleanSource ./.;
            CARGO_BUILD_TARGET = "wasm32-unknown-unknown";
            doCheck = false;
          };
          wasm-dir = pkgs.runCommand "gpuiua-dir" {} ''
            ${pkgs.wasm-bindgen-cli}/bin/wasm-bindgen ${wasm}/bin/gpuiua.wasm --target web --out-dir $out
            cp ${./index.html} $out/index.html
          '';
          serve = with pkgs; writeShellScriptBin "serve" "${caddy}/bin/caddy file-server --listen ':8000' --root ${wasm-dir}";
        };

        devShells.default = with pkgs;
          mkShell rec {
            nativeBuildInputs = [
              pkg-config
              linuxPackages_latest.perf
                            hotspot
            ];
            buildInputs = [
              udev
              alsa-lib
              vulkan-loader
              xorg.libX11
              xorg.libXcursor
              xorg.libXi
              xorg.libXrandr # To use the x11 feature
              libxkbcommon
              wayland # To use the wayland feature
            ];
            LD_LIBRARY_PATH = lib.makeLibraryPath buildInputs;
          };
      }
    );
}
