#!/usr/bin/env python3
"""
build_native.py — Cross-Platform Native Compiler for NeonRunnerBridgeV2 (v2.1.0)
================================================================================
Compiles `neon_stream_accelerator_v2.c` into:
  - Linux / ARMv8-A / macOS: `libneon_stream_accelerator.so` (+ `libtrinity_gemv.so`)
  - Windows x64: `neon_stream_accelerator.dll` (via GCC/Clang, full MSVC+SDK, or
    zero-SDK freestanding MSVC x64 AVX2 toolchain using `cl.exe` + `lib.exe`).
"""

import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _find_msvc_vc_dir() -> Path:
    roots = [
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC"),
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC"),
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC"),
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC"),
    ]
    for r in roots:
        if r.is_dir():
            versions = sorted([d for d in r.iterdir() if d.is_dir()])
            if versions:
                return versions[-1]
    raise RuntimeError("MSVC 2022 toolchain directory not found.")


def _has_windows_sdk_ucrt() -> bool:
    kits = Path(r"C:\Program Files (x86)\Windows Kits\10\Include")
    return kits.is_dir() and any(kits.rglob("stdio.h"))


def build_c_shared_library(
    c_src: Path,
    out_path: Path,
    defines: list = None,
    kernel32_exports: list = None,
    exports: list = None,
    use_avx2: bool = True,
) -> Path:
    """Compile a C file into a shared library (.dll on Windows, .so/.dylib on POSIX)."""
    c_src = Path(c_src).resolve()
    out_path = Path(out_path).resolve()
    base = c_src.parent
    defines = list(defines or [])

    if sys.platform == "win32":
        gcc = shutil.which("gcc") or shutil.which("clang")
        if gcc:
            flags = [gcc, "-O3", *(["-mavx2", "-mfma"] if use_avx2 else []), "-fopenmp"]
            flags.extend(f"-D{d}" for d in defines)
            flags.extend(["-shared", str(c_src), "-o", str(out_path)])
            subprocess.run(flags, cwd=str(base), check=True)
            return out_path

        vc_dir = _find_msvc_vc_dir()
        bin_dir = vc_dir / "bin" / "Hostx64" / "x64"
        inc_dir = vc_dir / "include"
        lib_dir = vc_dir / "lib" / "x64"
        cl_exe = bin_dir / "cl.exe"
        lib_exe = bin_dir / "lib.exe"

        env = os.environ.copy()
        env["PATH"] = f"{bin_dir};{env.get('PATH', '')}"

        if _has_windows_sdk_ucrt():
            devshell = vc_dir.parents[2] / "Common7" / "Tools" / "Launch-VsDevShell.ps1"
            def_flags = " ".join(f"/D{d}" for d in defines)
            arch_flag = "/arch:AVX2" if use_avx2 else ""
            ps_cmd = (
                f"& '{devshell}' -Arch amd64 -SkipAutomaticLocation; "
                f"Set-Location '{base}'; "
                f"cl.exe /nologo /O2 {arch_flag} /openmp {def_flags} "
                f"/LD '{c_src}' /Fe:'{out_path}'; exit $LASTEXITCODE"
            )
            subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd], check=True)
            return out_path

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            def_file = tmp_path / "kernel32.def"
            k32_lib = tmp_path / "kernel32_min.lib"
            obj_file = tmp_path / "target.obj"
            (tmp_path / "stddef.h").write_text(
                "#pragma once\n"
                "#ifndef _SIZE_T_DEFINED\n"
                "#define _SIZE_T_DEFINED\n"
                "typedef unsigned __int64 size_t;\n"
                "#endif\n"
                "typedef __int64 ptrdiff_t;\n"
                "#ifndef NULL\n#define NULL ((void*)0)\n#endif\n",
                encoding="utf-8",
            )
            (tmp_path / "malloc.h").write_text("#pragma once\n#include <stddef.h>\n", encoding="utf-8")
            (tmp_path / "string.h").write_text("#pragma once\n#include <vcruntime_string.h>\n", encoding="utf-8")

            k32_list = list(kernel32_exports or ["QueryPerformanceFrequency", "QueryPerformanceCounter", "GetProcessHeap", "HeapAlloc", "HeapFree"])
            def_file.write_text("LIBRARY KERNEL32.dll\nEXPORTS\n" + "".join(f"  {sym}\n" for sym in k32_list), encoding="utf-8")
            subprocess.run(
                [str(lib_exe), "/NOLOGO", f"/DEF:{def_file}", "/MACHINE:X64", f"/OUT:{k32_lib}"],
                env=env,
                check=True,
            )
            chkstk_obj = lib_dir / "chkstk.obj"
            cl_flags = [
                str(cl_exe), "/nologo", "/O2", *(["/arch:AVX2"] if use_avx2 else []), "/openmp", "/GS-", "/Zl",
                f"/I{inc_dir}", f"/I{tmp_path}", f"/I{base}",
                *(f"/D{d}" for d in defines),
                "/DNEON_MSVC_FREESTANDING_DLL",
                "/c", str(c_src), f"/Fo{obj_file}",
            ]
            if not chkstk_obj.exists():
                cl_flags.insert(5, "/Gs1048576")
            subprocess.run(cl_flags, env=env, check=True)
            link_exe = bin_dir / "link.exe"
            link_inputs = [str(obj_file), str(k32_lib)]
            if chkstk_obj.exists():
                link_inputs.append(str(chkstk_obj))
            link_inputs.extend(["vcruntime.lib", "vcomp.lib"])
            link_inputs.extend(f"/EXPORT:{sym}" for sym in (exports or []))
            subprocess.run(
                [
                    str(link_exe), "/NOLOGO", "/DLL", "/NOENTRY", "/NODEFAULTLIB",
                    f"/LIBPATH:{lib_dir}",
                    *link_inputs,
                    f"/OUT:{out_path}",
                    f"/IMPLIB:{tmp_path / 'target.lib'}",
                ],
                env=env,
                check=True,
            )
        return out_path

    cc = os.environ.get("CC") or shutil.which("gcc") or shutil.which("clang")
    if not cc:
        raise RuntimeError("No C compiler (gcc/clang) found on POSIX host.")
    machine = platform.machine().lower()
    if machine == "aarch64":
        arch_flags = ["-march=armv8-a+simd", "-mcpu=cortex-a72"]
    elif machine in ("x86_64", "amd64") and use_avx2:
        arch_flags = ["-mavx2", "-mfma"]
    else:
        arch_flags = []
    omp_flags = ["-fopenmp"] if sys.platform != "darwin" else []
    subprocess.run(
        [cc, "-O3", "-std=c11", *arch_flags, *omp_flags, *(f"-D{d}" for d in defines), "-shared", "-fPIC", str(c_src), "-o", str(out_path), "-lm"],
        cwd=str(base),
        check=True,
    )
    return out_path


def build_shared_library(src_dir: Path = None) -> Path:
    base = (src_dir or Path(__file__).resolve().parent).resolve()
    c_src = base / "neon_stream_accelerator_v2.c"
    if not c_src.exists():
        raise FileNotFoundError(f"Missing C source file: {c_src}")

    if sys.platform == "win32":
        out_dll = base / "neon_stream_accelerator.dll"
        build_c_shared_library(
            c_src,
            out_dll,
            defines=["NEON_BUILDING_DLL", "NEON_SHARED_LIB_ONLY"],
            use_avx2=True,
        )
        for ext in (".obj", ".exp", ".lib"):
            for f in base.glob(f"*{ext}"):
                f.unlink(missing_ok=True)
        return out_dll

    # POSIX (Linux ARMv8-A NEON, Linux x86_64, macOS)
    cc = os.environ.get("CC") or shutil.which("gcc") or shutil.which("clang")
    if not cc:
        raise RuntimeError("No C compiler (gcc/clang) found on POSIX host.")

    machine = platform.machine().lower()
    if machine == "aarch64":
        arch_flags = ["-march=armv8-a+simd", "-mcpu=cortex-a72"]
    elif machine in ("x86_64", "amd64"):
        arch_flags = ["-mavx2", "-mfma"]
    else:
        arch_flags = []
    omp_flags = ["-fopenmp"] if sys.platform != "darwin" else []
    ext = ".dylib" if sys.platform == "darwin" else ".so"
    out_so = base / f"libneon_stream_accelerator{ext}"
    compat_so = base / f"libtrinity_gemv{ext}"
    out_bin = base / "neon_stream_accelerator_v2"

    build_c_shared_library(c_src, out_so, defines=["NEON_SHARED_LIB_ONLY"], use_avx2=True)
    shutil.copy2(out_so, compat_so)
    subprocess.run(
        [cc, "-O3", *arch_flags, *omp_flags, str(c_src), "-o", str(out_bin), "-lm"],
        cwd=str(base),
        check=True,
    )
    return out_so


if __name__ == "__main__":
    built = build_shared_library()
    print(f"[+] Built native NeonRunnerBridgeV2 library: {built} ({built.stat().st_size:,} bytes)")
