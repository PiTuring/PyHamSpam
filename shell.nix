{ pkgs ? import <nixpkgs> {} }:

with pkgs.python313Packages;

pkgs.mkShell {
  buildInputs = with pkgs; [
    numpy
  ];
}
