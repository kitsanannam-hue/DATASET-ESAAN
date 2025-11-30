
{ pkgs }: {
  deps = [
    pkgs.python311
    pkgs.python311Packages.pip
    pkgs.python311Packages.torch
    pkgs.python311Packages.numpy
    pkgs.python311Packages.scipy
    pkgs.libsndfile
    pkgs.ffmpeg
  ];
}
