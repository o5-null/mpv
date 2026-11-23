#!/usr/bin/env python3
"""Package the locally built Windows mpv into a flat, self-contained zip.

This is the packaging half of the "our own mpv build" pipeline: the MSYS2
CLANG64 build is *dynamically* linked against a pile of DLLs under
``C:\\msys64\\clang64\\bin``. Instead of shipping a launcher script or asking
the user to install MSYS2, we walk the import table closure with objdump and
flatten every non-system DLL next to mpv.exe inside the zip.

Usage:
    python scripts/package_windows.py \
        --out work/mpv-x86_64-playkit-2026.11.23.zip --version 2026.11.23

Only the Python standard library is used, so this runs in plain Windows Python
(no MSYS2 environment required) as well as inside the MSYS2 shell.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

# Reproducible archives: every entry gets the same timestamp.
FIXED_DATE_TIME = (2020, 1, 1, 0, 0, 0)

# Never followed even if a copy happens to sit in the msys bin dir, and never
# worth reporting as "missing" (they are part of Windows itself).
SYSTEM_PREFIXES = (
    "api-ms-win-",
    "ext-ms-win-",
)
SYSTEM_DLLS = {
    "advapi32.dll",
    "avicap32.dll",
    "avrt.dll",
    "bcryptprimitives.dll",
    "dnsapi.dll",
    "dwrite.dll",
    "bcrypt.dll",
    "cfgmgr32.dll",
    "combase.dll",
    "comctl32.dll",
    "comdlg32.dll",
    "crypt32.dll",
    "d3d9.dll",
    "d3d11.dll",
    "d3d12.dll",
    "dbghelp.dll",
    "dwmapi.dll",
    "dxgi.dll",
    "gdi32.dll",
    "gdiplus.dll",
    "imm32.dll",
    "iphlpapi.dll",
    "kernel32.dll",
    "mpr.dll",
    "msimg32.dll",
    "msvcrt.dll",
    "ncrypt.dll",
    "normaliz.dll",
    "ntdll.dll",
    "ole32.dll",
    "oleaut32.dll",
    "opengl32.dll",
    "powrprof.dll",
    "psapi.dll",
    "rpcrt4.dll",
    "secur32.dll",
    "setupapi.dll",
    "shell32.dll",
    "shlwapi.dll",
    "shcore.dll",
    "user32.dll",
    "userenv.dll",
    "usp10.dll",
    "uxtheme.dll",
    "version.dll",
    "winhttp.dll",
    "wininet.dll",
    "winmm.dll",
    "wldap32.dll",
    "ws2_32.dll",
    "wsock32.dll",
    "wtsapi32.dll",
    "xaudio2_9.dll",
}

DLL_RE = re.compile(r"DLL Name:\s*(\S+)", re.IGNORECASE)


def log(msg: str) -> None:
    print(msg, flush=True)


def is_system_dll(name: str) -> bool:
    lower = name.lower()
    return lower in SYSTEM_DLLS or lower.startswith(SYSTEM_PREFIXES)


def find_objdump(msys_bin: Path) -> Path:
    """objdump lives in the same bin dir as the DLLs; fall back to the known path."""
    for candidate in (
        msys_bin / "objdump.exe",
        msys_bin / "llvm-objdump.exe",
        Path(r"C:\msys64\clang64\bin\objdump.exe"),
    ):
        if candidate.is_file():
            return candidate
    raise SystemExit(
        f"error: objdump.exe not found (looked in {msys_bin} and the default MSYS2 path)",
    )


def read_dll_imports(objdump: Path, path: Path) -> list[str]:
    """Return the DLL names in the *static* import table of *path*.

    objdump prints several tables with a "DLL Name:" line each (import, export,
    delay import). Only the regular import table creates a real load-time
    dependency, so the output is parsed section by section: the export table's
    own name and delay-loaded (optional) libraries must not be bundled.
    """
    try:
        proc = subprocess.run(
            [str(objdump), "-p", str(path)],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log(f"  warning: objdump failed on {path.name}: {exc}")
        return []
    if proc.returncode != 0:
        log(f"  warning: objdump exit {proc.returncode} on {path.name}")
        return []
    # LLVM objdump writes to stdout, some builds to stderr; scan both.
    text = proc.stdout + "\n" + proc.stderr

    names: list[str] = []
    in_imports = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("The Import Tables") or stripped.startswith("Import Table:"):
            in_imports = True
            continue
        # Any other unindented section header closes the import table.
        if in_imports and line[:1] not in ("", " ", "\t"):
            in_imports = False
            continue
        if in_imports:
            match = DLL_RE.match(stripped)
            if match:
                names.append(match.group(1))
    # De-duplicate while keeping the order stable.
    return list(dict.fromkeys(names))



class Resolver:
    """Finds a DLL by name, build dir first, then the MSYS2 bin dir."""

    def __init__(self, build_dir: Path, msys_bin: Path) -> None:
        self.build_dir = build_dir
        self.msys_bin = msys_bin
        # Case-insensitive index of the msys bin dir (Windows paths are).
        self._msys_index = {
            entry.name.lower(): entry
            for entry in msys_bin.iterdir()
            if entry.is_file()
        }

    def resolve(self, name: str) -> Path | None:
        local = self.build_dir / name
        if local.is_file():
            return local
        return self._msys_index.get(name.lower())

    def search_dirs(self) -> tuple[Path, Path]:
        return self.build_dir, self.msys_bin


def build_closure(
    seeds: list[Path], resolver: Resolver, objdump: Path,
) -> tuple[dict[str, Path], list[str]]:
    """Walk the import graph breadth-first until it stops growing.

    Returns (dll name -> source path, sorted list of dependency names that could
    not be satisfied and were therefore skipped).
    """
    closure: dict[str, Path] = {}
    skipped: set[str] = set()
    queue = list(seeds)
    seen_files: set[Path] = set()

    while queue:
        current = queue.pop(0)
        if current in seen_files:
            continue
        seen_files.add(current)
        for dep in read_dll_imports(objdump, current):
            if is_system_dll(dep):
                continue
            key = dep.lower()
            if key in closure:
                continue
            source = resolver.resolve(dep)
            if source is None:
                skipped.add(dep)
                continue
            closure[key] = source
            queue.append(source)

    return closure, sorted(skipped)


def zipinfo(name: str) -> zipfile.ZipInfo:
    """A ZipInfo with a fixed timestamp and sane permissions (0644)."""
    info = zipfile.ZipInfo(filename=name, date_time=FIXED_DATE_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    info.create_system = 0  # MS-DOS, keeps extractors happy
    return info


def main(argv: list[str] | None = None) -> int:
    script_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Package the local CLANG64 mpv build into a flat zip.",
    )
    parser.add_argument(
        "--build-dir",
        default=str(script_root / "build-msys"),
        help="meson build directory holding mpv.exe (default: <repo>/build-msys)",
    )
    parser.add_argument(
        "--msys-bin",
        default=r"C:\msys64\clang64\bin",
        help=r"MSYS2 CLANG64 bin dir with the runtime DLLs (default: C:\msys64\clang64\bin)",
    )
    parser.add_argument("--out", required=True, help="output .zip path")
    parser.add_argument(
        "--version", default="", help="version string, used for the archive comment",
    )
    args = parser.parse_args(argv)

    def abspath(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (Path.cwd() / path).resolve()

    build_dir = abspath(args.build_dir)
    msys_bin = abspath(args.msys_bin)
    out = abspath(args.out)

    if not build_dir.is_dir():
        raise SystemExit(f"error: build dir not found: {build_dir}")
    if not msys_bin.is_dir():
        raise SystemExit(f"error: msys bin dir not found: {msys_bin}")

    objdump = find_objdump(msys_bin)
    log(f"build dir : {build_dir}")
    log(f"msys bin  : {msys_bin}")
    log(f"objdump   : {objdump}")

    # Seed files: the launcher pair plus libmpv-2.dll (used by third-party
    # players/frontends), and d3dcompiler_43 if MSYS2 ships it.
    seeds: list[Path] = []
    for exe in ("mpv.exe", "mpv.com", "libmpv-2.dll"):
        path = build_dir / exe
        if not path.is_file():
            raise SystemExit(f"error: missing {path} (rebuild with -Dlibmpv=true)")
        seeds.append(path)
        log(f"seed      : {exe} ({path.stat().st_size:,} bytes)")

    payload: dict[str, Path] = {p.name: p for p in seeds}

    d3dcompiler = msys_bin / "d3dcompiler_43.dll"
    if d3dcompiler.is_file():
        payload[d3dcompiler.name] = d3dcompiler
        log(f"seed      : {d3dcompiler.name} ({d3dcompiler.stat().st_size:,} bytes)")
    else:
        log("note      : d3dcompiler_43.dll not found in msys bin, skipping")

    resolver = Resolver(build_dir, msys_bin)
    closure, skipped = build_closure(seeds, resolver, objdump)

    for key in sorted(closure):
        source = closure[key]
        name = source.name
        if name in payload:
            continue
        payload[name] = source

    # Deterministic entry order, and never embed a directory.
    entries = sorted(payload.items(), key=lambda item: item[0].lower())

    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="mpv-package-"))
    try:
        for name, source in entries:
            shutil.copy2(source, staging / name)

        total = 0
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
            if args.version:
                zf.comment = (
                    f"mpv-playkit {args.version} (self-built, dynamic deps bundled)".encode()
                )
            for name, _source in entries:
                staged = staging / name
                total += staged.stat().st_size
                zf.writestr(zipinfo(name), staged.read_bytes(), zipfile.ZIP_DEFLATED)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    zip_size = out.stat().st_size
    log("")
    log(f"entries   : {len(entries)} files")
    log(f"raw size  : {total:,} bytes ({total / (1024 * 1024):.1f} MiB)")
    log(f"zip size  : {zip_size:,} bytes ({zip_size / (1024 * 1024):.1f} MiB)")
    log(f"archive   : {out}")
    if skipped:
        suffix = "y" if len(skipped) == 1 else "ies"
        log(f"skipped {len(skipped)} unresolved/unwanted dependenc{suffix}:")
        for name in skipped:
            log(f"  - {name}")
    else:
        log("skipped   : none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
