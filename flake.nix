{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    systems.url = "github:nix-systems/default";
    devenv.url = "github:cachix/devenv";
    nixpkgs-python = {
      url = "github:cachix/nixpkgs-python";
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
        pwd = builtins.getEnv "PWD";
        platforms = builtins.toPath "${pwd}/../qibolab_platforms_qrc/";
      in {
        default = devenv.lib.mkShell {
          inherit inputs pkgs;

          modules = [
            ({lib, ...}: {
              packages = with pkgs; [pre-commit poethepoet jupyter];

              env.QIBOLAB_PLATFORMS = platforms;

              languages.python = {
                enable = true;
                poetry = {
                  enable = true;
                  install = {
                    enable = true;
                    groups = ["dev" "tests"];
                    allExtras = true;
                  };
                };
              };
            })
          ];
        };
      });
  };
}
