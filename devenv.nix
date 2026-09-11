{ pkgs, ... }:
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
    venv.enable = true;
    version = "3.12";
    uv = {
      enable = true;
      sync = {
        enable = true;
        allGroups = true;
        allExtras = true;
      };
    };
  };
}
