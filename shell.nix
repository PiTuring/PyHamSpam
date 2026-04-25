{ pkgs ? import <nixpkgs> {} }:

with pkgs.python313Packages;

let
  ty-check = pkgs.writeShellScriptBin "pyrun" ''
    ty check --color never "$@" && python "$@"
  '';
in pkgs.mkShell {
  buildInputs = with pkgs; [
    numpy
    ty
    ty-check
    matplotlib
  ];
  TY_OUTPUT_FORMAT = "concise";
}
