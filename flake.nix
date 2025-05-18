{
  description = "Flake utils demo";

  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs = {
    self,
    flake-utils,
    nixpkgs,
  }:
    flake-utils.lib.eachDefaultSystem (
      system: {
        devShells.default = with nixpkgs.legacyPackages.${system};
          mkShell rec {
            nativeBuildInputs = [
              pkg-config
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
