{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };
  outputs = {nixpkgs, ...}: let
    inherit (nixpkgs) lib;
    withSystem = f:
      lib.fold lib.recursiveUpdate {}
      (map f ["x86_64-linux" "x86_64-darwin" "aarch64-linux" "aarch64-darwin"]);
  in
    withSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
        inherit (pkgs) stdenv;
        # Use impure cc on Darwin so that python packages build correctly
        mkShell =
          if stdenv.isLinux
          then pkgs.mkShell
          else pkgs.mkShell.override {stdenv = pkgs.stdenvNoCC;};
      in {
        devShells.${system}.default =
          mkShell
          {
            packages = with pkgs; [
              python313
              uv
              julia-bin
            ];

            # Use ipdb
            PYTHONBREAKPOINT = "ipdb.set_trace";
            shellHook = ''
              if [ ! -d .venv ]; then
                uv venv --python 'python3.13'
                uv pip install -e .
                uv sync --all-extras --all-groups
                if [ -f requirements.txt ]; then
                  uv pip sync requirements.txt
                fi
              fi
              source .venv/bin/activate
            '';
          };
      }
    );
}
