{ lib, pkgs, ... }:
{
  packages = with pkgs; [
    pre-commit
    poethepoet
    jupyter
  ];

  env = {
    QIBOLAB_PLATFORMS = (dirOf (toString ./.)) + "/qibolab_platforms_qrc";
    PYTHONBREAKPOINT = "pudb.set_trace";
  };

  languages.python = {
    enable = true;
    libraries = with pkgs; [ zlib ];
    version = "3.12";
    poetry = {
      enable = true;
      install = {
        enable = true;
        groups = [
          "dev"
          "analysis"
          "tests"
        ];
        extras = [
          (lib.strings.concatStrings (
            lib.strings.intersperse " -E " [
              "qrng"
              "emulator"
            ]
          ))
        ];
      };
    };
  };
}
