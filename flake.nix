{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    systems.url = "github:nix-systems/default";
    devenv.url = "github:cachix/devenv";
    nixpkgs-python = {
      url = "github:cachix/nixpkgs-python";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {
    self,
    nixpkgs,
    devenv,
    systems,
    ...
  } @ inputs: let
    forEachSystem = nixpkgs.lib.genAttrs (import systems);
  in {
    packages = forEachSystem (system: {
      default =
        nixpkgs.legacyPackages.${system}.poetry2nix.mkPoetryApplication
        {
          projectDir = self;
          preferWheels = true;
        };
    });

    devShells =
      forEachSystem
      (system: let
        pkgs = nixpkgs.legacyPackages.${system};
        lib = pkgs.lib;
        isDarwin = lib.strings.hasSuffix "darwin" system;
      in {
        default = devenv.lib.mkShell {
          inherit inputs pkgs;

          modules = [
            ({
              pkgs,
              config,
              ...
            }: {
              packages = with pkgs; [pre-commit poethepoet jupyter zlib] ++ lib.optionals isDarwin [stdenv.cc.cc.lib];

              env = {
                QIBOLAB_PLATFORMS = (dirOf config.env.DEVENV_ROOT) + "/qibolab_platforms_qrc";
              };

              languages.c = {
                enable = true;
              };

              languages.cplusplus = {
                enable = true;
              };

              languages.python = {
                enable = true;
                poetry = {
                  enable = true;
                  install.enable = true;
                  install.groups = ["dev" "tests"];
                  install.allExtras = true;
                };
                version = "3.11";
              };

              languages.rust = {
                enable = true;
                channel = "stable";
              };
            })
          ];
        };
      });
  };

  nixConfig = {
    extra-trusted-public-keys = "devenv.cachix.org-1:w1cLUi8dv3hnoSPGAuibQv+f9TZLr6cv/Hm9XgU50cw=";
    extra-substituters = "https://devenv.cachix.org";
  };
}
