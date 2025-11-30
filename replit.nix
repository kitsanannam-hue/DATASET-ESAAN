
{ pkgs }: {
  deps = [
    pkgs.python311
    pkgs.python311Packages.pip
    pkgs.python311Packages.torch
    pkgs.python311Packages.torchaudio
    pkgs.python311Packages.numpy
    pkgs.python311Packages.scipy
    pkgs.python311Packages.librosa
    pkgs.python311Packages.soundfile
    pkgs.python311Packages.tqdm
    pkgs.python311Packages.einops
    pkgs.python311Packages.tensorboard
    pkgs.python311Packages.pytest
    pkgs.python311Packages.pytest-cov
    pkgs.python311Packages.black
    pkgs.python311Packages.flake8
    pkgs.python311Packages.mypy
    pkgs.libsndfile
    pkgs.ffmpeg
  ];
  env = {
    PYTHONBIN = "${pkgs.python311}/bin/python3.11";
    LANG = "en_US.UTF-8";
    PIP_DISABLE_PIP_VERSION_CHECK = "1";
  };
}
