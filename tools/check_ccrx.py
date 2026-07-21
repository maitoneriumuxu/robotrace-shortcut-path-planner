from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import tempfile


DEFAULT_COMPILER = Path(
    r"C:\Program Files (x86)\Renesas Electronics\CS+\CC\CC-RX\V3.07.00\bin\ccrx.exe"
)


def main() -> int:
    parser = argparse.ArgumentParser(description="RX651向けC実装のCC-RX静的確認")
    parser.add_argument("--compiler", type=Path, default=DEFAULT_COMPILER)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / "embedded/shortcut_curvature_limiter.c"

    with tempfile.TemporaryDirectory() as temporary_text:
        temporary = Path(temporary_text)
        object_file = temporary / "limiter.obj"
        assembly_file = temporary / "limiter.src"
        common = [
            str(args.compiler),
            "-isa=rxv2",
            "-fpu",
            "-include=" + str(root / "embedded"),
        ]
        subprocess.run(
            common + ["-output=obj=" + str(object_file), str(source)],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            common + ["-output=src=" + str(assembly_file), str(source)],
            check=True,
            capture_output=True,
            text=True,
        )
        object_bytes = object_file.stat().st_size
        assembly = assembly_file.read_text(encoding="cp932")
        fsqrt_count = assembly.count("FSQRT")
        software_sqrt_call = "_$sqrtf" in assembly or "_sqrtf" in assembly

    print(
        f"CC-RX: object={object_bytes}byte work<=5160byte "
        f"FSQRT={fsqrt_count} software_sqrt_call={software_sqrt_call}"
    )
    if object_bytes > 16_384 or fsqrt_count == 0 or software_sqrt_call:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
