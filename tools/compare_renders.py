from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create side-by-side comparisons for matching render outputs."
    )
    parser.add_argument("--left-dir", default=None, help="Folder for the baseline run.")
    parser.add_argument("--right-dir", default=None, help="Folder for the comparison run.")
    parser.add_argument("--single-run-dir", default=None, help="One output folder containing both mesh-prior and completion images.")
    parser.add_argument("--output-dir", required=True, help="Folder where side-by-side comparison images will be written.")
    parser.add_argument(
        "--left-label",
        default="Mesh Prior",
        help="Label drawn above the left image.",
    )
    parser.add_argument(
        "--right-label",
        default="With Completion",
        help="Label drawn above the right image.",
    )
    parser.add_argument(
        "--pattern",
        default="view_*_render.png",
        help="Glob pattern used to find matching render images.",
    )
    parser.add_argument(
        "--contact-sheet",
        action="store_true",
        help="Also build one contact sheet containing every comparison pair.",
    )
    args = parser.parse_args()
    if bool(args.single_run_dir) == bool(args.left_dir or args.right_dir):
        raise SystemExit("Use either --single-run-dir or both --left-dir and --right-dir.")
    if args.single_run_dir is None and (args.left_dir is None or args.right_dir is None):
        raise SystemExit("When not using --single-run-dir, both --left-dir and --right-dir are required.")
    return args


def load_font() -> ImageFont.ImageFont:
    # Use a common system font when available, then fall back gracefully so the
    # comparison tool still works on minimal Python installations.
    try:
        return ImageFont.truetype("arial.ttf", 22)
    except OSError:
        return ImageFont.load_default()


def draw_label(draw: ImageDraw.ImageDraw, text: str, x: int, y: int, font: ImageFont.ImageFont) -> None:
    draw.text((x + 1, y + 1), text, fill=(255, 255, 255), font=font)
    draw.text((x, y), text, fill=(35, 43, 58), font=font)


def build_side_by_side(
    left_image: Image.Image,
    right_image: Image.Image,
    left_label: str,
    right_label: str,
    font: ImageFont.ImageFont,
) -> Image.Image:
    # Each comparison image keeps the layout intentionally simple: same-size
    # images, one label band, and no resizing tricks beyond matching the right
    # image to the left if their dimensions differ.
    label_band = 42
    gap = 12
    width = left_image.width + right_image.width + gap
    height = max(left_image.height, right_image.height) + label_band
    canvas = Image.new("RGB", (width, height), color=(247, 249, 252))
    canvas.paste(left_image, (0, label_band))
    canvas.paste(right_image, (left_image.width + gap, label_band))

    draw = ImageDraw.Draw(canvas)
    draw_label(draw, left_label, 10, 10, font)
    draw_label(draw, right_label, left_image.width + gap + 10, 10, font)
    return canvas


def iter_matching_files(left_dir: Path, right_dir: Path, pattern: str) -> list[tuple[Path, Path]]:
    # Two-run mode: match images by filename so separate experiment folders can
    # still be compared directly.
    right_lookup = {path.name: path for path in right_dir.glob(pattern)}
    pairs: list[tuple[Path, Path]] = []
    for left_path in sorted(left_dir.glob(pattern)):
        right_path = right_lookup.get(left_path.name)
        if right_path is not None:
            pairs.append((left_path, right_path))
    return pairs


def iter_single_run_pairs(run_dir: Path) -> list[tuple[Path, Path]]:
    # A single training run now writes both mesh-prior-only and with-completion
    # renders. Pair them by view index so comparison images can be generated
    # without managing two separate output folders.
    right_lookup = {
        path.name.replace("_with_completion", ""): path
        for path in run_dir.glob("view_*_with_completion.png")
    }
    pairs: list[tuple[Path, Path]] = []
    for left_path in sorted(run_dir.glob("view_*_mesh_prior.png")):
        key = left_path.name.replace("_mesh_prior", "")
        right_path = right_lookup.get(key)
        if right_path is not None:
            pairs.append((left_path, right_path))
    return pairs


def build_contact_sheet(images: list[Image.Image]) -> Image.Image:
    # Contact sheets are useful for reports because they give one file showing
    # every compared view instead of a directory full of per-view images.
    if not images:
        raise ValueError("No images were provided for the contact sheet.")

    gap = 18
    rows = len(images)
    width = max(image.width for image in images)
    height = sum(image.height for image in images) + gap * (rows - 1)
    sheet = Image.new("RGB", (width, height), color=(255, 255, 255))

    y = 0
    for image in images:
        sheet.paste(image, (0, y))
        y += image.height + gap
    return sheet


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.single_run_dir is not None:
        run_dir = Path(args.single_run_dir)
        pairs = iter_single_run_pairs(run_dir)
    else:
        left_dir = Path(args.left_dir)
        right_dir = Path(args.right_dir)
        pairs = iter_matching_files(left_dir, right_dir, args.pattern)
    if not pairs:
        if args.single_run_dir is not None:
            raise SystemExit(f"No matching mesh-prior / with-completion image pairs found in {run_dir}.")
        raise SystemExit(f"No matching files found for pattern '{args.pattern}' between {left_dir} and {right_dir}.")

    font = load_font()
    comparison_images: list[Image.Image] = []

    for left_path, right_path in pairs:
        left_image = Image.open(left_path).convert("RGB")
        right_image = Image.open(right_path).convert("RGB")

        if left_image.size != right_image.size:
            right_image = right_image.resize(left_image.size, Image.Resampling.LANCZOS)

        comparison = build_side_by_side(
            left_image=left_image,
            right_image=right_image,
            left_label=args.left_label,
            right_label=args.right_label,
            font=font,
        )
        comparison_path = output_dir / left_path.name.replace("_render", "_comparison")
        comparison.save(comparison_path)
        comparison_images.append(comparison)

    if args.contact_sheet:
        contact_sheet = build_contact_sheet(comparison_images)
        contact_sheet.save(output_dir / "comparison_contact_sheet.png")

    print(f"Wrote {len(comparison_images)} comparison image(s) to: {output_dir}")


if __name__ == "__main__":
    main()
