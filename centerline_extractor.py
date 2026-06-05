#!/usr/bin/env python3
"""Run the stroke-oriented centerline extractor on all sample PNGs."""

from pathlib import Path

from skeleton import process_geometric_centerline


def extract_centerline(png_path: str, svg_path: str) -> None:
    """Extract one PNG centerline SVG using the cleaned stroke pipeline."""
    output_prefix = str(Path(svg_path).with_suffix(""))
    process_geometric_centerline(png_path, output_prefix, debug=False)

    generated_svg = Path(f"{output_prefix}_fixed.svg")
    target_svg = Path(svg_path)
    if generated_svg != target_svg:
        target_svg.write_text(generated_svg.read_text())
        generated_svg.unlink()


def main() -> None:
    sample_dir = Path("challenge_sample")
    output_dir = Path("challenge_sample_results")
    output_dir.mkdir(exist_ok=True)

    png_files = sorted(sample_dir.glob("*.png"))
    print(f"Found {len(png_files)} PNG files")

    for png_file in png_files:
        svg_file = output_dir / f"{png_file.stem}.svg"
        print(f"Processing {png_file.name}...")
        extract_centerline(str(png_file), str(svg_file))
        print(f"  -> {svg_file}")


if __name__ == "__main__":
    main()
