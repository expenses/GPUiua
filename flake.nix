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
          wasm = wasm-crane-lib.buildTrunkPackage {
            name = "gpuiua";
            src = pkgs.lib.cleanSource ./.;
            RUSTFLAGS = ''--cfg getrandom_backend="wasm_js"'';
            trunkExtraBuildArgs = ''--public-url https://expenses.github.io/GPUiua/'';
          };
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
