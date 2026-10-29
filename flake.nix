{
  description = "open-ant";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };
      in
        {
          devShells.default = pkgs.mkShell {
            buildInputs = with pkgs; [
              circup
              picocom
              python312
              python312Packages.venvShellHook
            ];

            venvDir = "./.venv";

            postVenvCreation = ''
              unset SOURCE_DATE_EPOCH
              pip install --upgrade pip
              pip install -r requirements.txt
            '';

            shellHook = ''
              alias circpush="rsync -avh --progress --fsync code.py /run/media/$(whoami)/CIRCUITPY/"
            '';
          };
        }
    );
}
