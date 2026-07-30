{
  description = "Bonsai-27B on Vega 56/64 (gfx900): reproducible PrismML llama.cpp builds (Vulkan, later ROCm) + pinned RADV stack for the rig";

  inputs = {
    # Full rev (from channels.nixos.org/nixpkgs-unstable) so nix fetches the
    # codeload tarball directly and never touches the rate-limited GitHub API.
    nixpkgs.url = "github:NixOS/nixpkgs/9bc02893134c733dd85de46ee4fb2fac696b5529";
    # Pinned to the exact release running on the rig: prism-b9596-9fcaed7
    prism-llamacpp = {
      url = "github:PrismML-Eng/llama.cpp/9fcaed763ccda38ea81068ad9d7f991aaddca451";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, prism-llamacpp }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};

      # Every file in patches/ (sorted) is applied on top of the pinned source.
      # Baseline build = empty patches dir.
      patchDir = ./patches;
      patchFiles =
        let entries = builtins.readDir patchDir;
        in map (n: patchDir + "/${n}")
          (builtins.sort builtins.lessThan
            (builtins.filter (n: nixpkgs.lib.hasSuffix ".patch" n)
              (builtins.attrNames entries)));

      prism-llama-vulkan = pkgs.stdenv.mkDerivation {
        pname = "prism-llama-vulkan";
        version = "b9596-9fcaed7${nixpkgs.lib.optionalString (patchFiles != [ ]) "-patched"}";
        src = prism-llamacpp;
        patches = patchFiles;

        nativeBuildInputs = with pkgs; [ cmake ninja pkg-config python3 shaderc.bin ];
        buildInputs = with pkgs; [ vulkan-headers vulkan-loader spirv-headers spirv-tools ];

        cmakeFlags = [
          # Mirrors .github/workflows/release-prism.yml (ubuntu-vulkan job)
          "-DGGML_VULKAN=ON"
          "-DGGML_BACKEND_DL=ON"
          "-DGGML_NATIVE=OFF"
          "-DGGML_CPU_ALL_VARIANTS=ON"
          "-DLLAMA_CURL=OFF"
          "-DLLAMA_BUILD_TESTS=OFF"
          "-DCMAKE_BUILD_TYPE=Release"
        ];

        # Nix strips by default; keep it, binaries are small anyway.
        doCheck = false;
      };

      # ROCm/HIP build for gfx900. nixpkgs ROCm 7.x still ships gfx900 targets.
      prism-llama-rocm = pkgs.stdenv.mkDerivation {
        pname = "prism-llama-rocm";
        version = "b9596-9fcaed7${nixpkgs.lib.optionalString (patchFiles != [ ]) "-patched"}";
        src = prism-llamacpp;
        patches = patchFiles;

        nativeBuildInputs = with pkgs; [ cmake ninja pkg-config perl rocmPackages.llvm.clang ];
        buildInputs = with pkgs.rocmPackages; [ clr rocblas hipblas rocwmma ]
          ++ [ pkgs.libffi ];

        cmakeFlags = [
          "-DGGML_HIP=ON"
          "-DAMDGPU_TARGETS=gfx900"
          "-DGPU_TARGETS=gfx900"
          "-DCMAKE_HIP_ARCHITECTURES=gfx900"
          "-DGGML_BACKEND_DL=ON"
          "-DGGML_NATIVE=OFF"
          # rig CPU is SSE4.2-only; without runtime-dispatched CPU variants the
          # default x86 build SIGILLs on startup
          "-DGGML_CPU_ALL_VARIANTS=ON"
          "-DLLAMA_CURL=OFF"
          "-DLLAMA_BUILD_TESTS=OFF"
          "-DCMAKE_BUILD_TYPE=Release"
          "-DCMAKE_HIP_COMPILER_ROCM_ROOT=${pkgs.rocmPackages.clr}"
        ];
      };

      # Wrapper env for the rig: pinned RADV (Mesa from this nixpkgs) instead of
      # the rig's Mesa 23.2.1. Everything the binary needs rides in the closure.
      vega-runtime = pkgs.runCommand "vega-runtime"
        {
          nativeBuildInputs = [ pkgs.makeWrapper ];
          mesa = pkgs.mesa;
          llama = prism-llama-vulkan;
        } ''
        icd=$(ls "$mesa"/share/vulkan/icd.d/radeon_icd.x86_64.json 2>/dev/null \
           || ls "${pkgs.mesa.drivers or pkgs.mesa}"/share/vulkan/icd.d/radeon_icd.x86_64.json 2>/dev/null) \
           || { echo "RADV ICD json not found in mesa outputs"; exit 1; }
        mkdir -p $out/bin
        for exe in "$llama"/bin/llama-*; do
          name=$(basename "$exe")
          makeWrapper "$exe" "$out/bin/$name" \
            --set VK_ICD_FILENAMES "$icd" \
            --set-default GGML_VK_VISIBLE_DEVICES 0
        done
        ln -s "$llama" $out/llama
        ln -s "$mesa" $out/mesa
      '';
    in
    {
      packages.${system} = {
        default = prism-llama-vulkan;
        inherit prism-llama-vulkan prism-llama-rocm vega-runtime;
      };

      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          cmake
          ninja
          pkg-config
          python3
          shaderc.bin
          vulkan-headers
          vulkan-loader
          vulkan-tools
          spirv-tools
          gdb
        ];
      };
    };
}
