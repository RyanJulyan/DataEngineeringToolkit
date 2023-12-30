{pkgs}: {
  deps = [
    pkgs.xsimd
    pkgs.pkg-config
    pkgs.libxcrypt
    pkgs.glibcLocales
    pkgs.glibc
  ];
  env = {
    PYTHON_LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
      pkgs.glibcLocales
    ];
  };
}
