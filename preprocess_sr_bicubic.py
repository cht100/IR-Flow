import argparse
import re
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
X_SUFFIX = re.compile(r"x(?:2|3|4|8)$", re.IGNORECASE)
BICUBIC = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC


def image_files(folder):
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def normalized_stem(path):
    return X_SUFFIX.sub("", path.stem)


def build_target_map(target_dir):
    mapping = {}
    for path in image_files(target_dir):
        key = normalized_stem(path)
        if key in mapping:
            raise RuntimeError(f"Duplicate target key '{key}': {mapping[key]} and {path}")
        mapping[key] = path
    return mapping


def process_split(root, split, input_name, target_name, output_name):
    split_dir = root / split
    input_dir = split_dir / input_name
    target_dir = split_dir / target_name
    output_dir = split_dir / output_name

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not target_dir.is_dir():
        raise FileNotFoundError(f"Target directory not found: {target_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    targets = build_target_map(target_dir)
    inputs = image_files(input_dir)
    if not inputs:
        raise RuntimeError(f"No images found in {input_dir}")

    processed = 0
    missing = []
    for index, input_path in enumerate(inputs, start=1):
        target_path = targets.get(normalized_stem(input_path))
        if target_path is None:
            missing.append(input_path.name)
            continue

        with Image.open(input_path) as lr_image, Image.open(target_path) as target_image:
            target_size = target_image.size
            lr_image = lr_image.convert("RGB")
            upscaled = lr_image.resize(target_size, BICUBIC)

            # Use the target filename so folder sorting pairs LQ and GT reliably.
            output_path = output_dir / target_path.name
            save_kwargs = {"quality": 95, "subsampling": 0} if output_path.suffix.lower() in {".jpg", ".jpeg"} else {}
            upscaled.save(output_path, **save_kwargs)

        processed += 1
        if index % 100 == 0 or index == len(inputs):
            print(f"[{split}] {index}/{len(inputs)} scanned, {processed} written")

    if missing:
        preview = ", ".join(missing[:10])
        raise RuntimeError(f"{len(missing)} inputs have no matching target. First entries: {preview}")

    if processed != len(targets):
        raise RuntimeError(
            f"Pair count mismatch in {split}: wrote {processed} inputs but found {len(targets)} targets"
        )
    print(f"[{split}] done: {processed} images -> {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bicubic-upscale SR inputs to each paired target's native size."
    )
    parser.add_argument("--data_root", required=True, help="Dataset root containing train/test folders.")
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--input_name", default="input")
    parser.add_argument("--target_name", default="target")
    parser.add_argument("--output_name", default="input_bicubic")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.data_root).expanduser().resolve()
    for split in args.splits:
        process_split(root, split, args.input_name, args.target_name, args.output_name)


if __name__ == "__main__":
    main()
