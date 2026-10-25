"""
Python rewrite of "Load tif into PSD.js".

Original principle (Photoshop):
  1. Open the PSD, open the matching TIF.
  2. Place the TIF as a new top layer on the PSD.
  3. Set that layer's blend mode to "Difference".
  4. Export the result.

Python equivalent (no Photoshop needed):
  1. Render the full PSD composite with psd-tools  -> base image.
  2. Since the TIF sits on TOP with Difference blend:
         final = | base - tif |
     which is exactly ImageChops.difference(base, tif).
  3. Save the result as PNG into <TIF folder>/PSD_check/.

Performance:
  - Fast compositor (PIL C ops) for PSDs made of simple pixel layers with
    NORMAL/SCREEN/MULTIPLY/DIFFERENCE blends (~3-4x faster than psd-tools).
  - Falls back to psd-tools' full composite for anything more complex
    (type layers, smart objects, masks, adjustments, other blend modes).
  - Files are processed in parallel with a thread pool.
  - Missing dependencies are auto-installed on startup.
"""

import argparse
import os
import subprocess
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# Startup self-check: auto-install missing libraries, then import them.
# ---------------------------------------------------------------------------
REQUIRED = {
    "PIL": "Pillow",
    "psd_tools": "psd-tools",
}


def ensure_dependencies() -> None:
    missing = []
    for module, package in REQUIRED.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"Missing libraries: {', '.join(missing)}. Installing...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", *missing]
            )
        except subprocess.CalledProcessError:
            sys.exit(f"Failed to install: {', '.join(missing)}. Install manually with:\n"
                     f"    {sys.executable} -m pip install {' '.join(missing)}")
        # Re-check after installation
        for module, package in REQUIRED.items():
            try:
                __import__(module)
            except ImportError:
                sys.exit(f"Library {package} was installed but still cannot be imported. "
                         "Please restart the script.")


ensure_dependencies()

from PIL import Image, ImageChops  # noqa: E402
from psd_tools import PSDImage  # noqa: E402
from psd_tools.constants import BlendMode  # noqa: E402

# Allow very large images (e.g. 10000+ px) without DecompressionBombError
Image.MAX_IMAGE_PIXELS = None

warnings.filterwarnings("ignore", category=UserWarning, module="psd_tools")

# Blend modes the fast compositor can handle, mapped to PIL C implementations.
_FAST_BLENDS = {
    BlendMode.NORMAL: None,               # handled as direct paste
    BlendMode.SCREEN: ImageChops.screen,
    BlendMode.MULTIPLY: ImageChops.multiply,
    BlendMode.DIFFERENCE: ImageChops.difference,
}


def _collect_simple_layers(layers):
    """Return a flat [(blend_mode, layer)] list if every visible layer is a
    plain pixel layer (or a plain pass-through/normal group of them), with
    full opacity, no clipping and no enabled masks. Otherwise return None."""
    flat = []
    for layer in layers:
        if not layer.visible or layer.opacity != 255:
            continue
        if layer.clipping is True:
            return None
        if layer.mask is not None and not layer.mask.disabled:
            return None
        if layer.kind == "pixel":
            if layer.blend_mode not in _FAST_BLENDS:
                return None
            flat.append((layer.blend_mode, layer))
        elif layer.kind == "group" and layer.blend_mode in (BlendMode.PASS_THROUGH, BlendMode.NORMAL):
            children = _collect_simple_layers(layer)
            if children is None:
                return None
            flat.extend(children)
        else:
            return None
    return flat or None


def _fast_composite(psd: PSDImage):
    """Composite simple PSDs with PIL C ops. Returns RGB Image or None if the
    document contains anything the fast path cannot handle."""
    flat = _collect_simple_layers(psd)
    if flat is None:
        return None

    canvas_w, canvas_h = psd.width, psd.height
    acc = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))  # white backdrop

    for mode, layer in flat:
        im = layer.topil()
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGBA")
        src_a = im.getchannel("A") if im.mode == "RGBA" else None
        opaque = src_a is None or src_a.getextrema() == (255, 255)

        left, top = layer.bbox[0], layer.bbox[1]  # bbox = (left, top, right, bottom)
        w, h = im.size
        l, t = max(0, left), max(0, top)
        r, b = min(canvas_w, left + w), min(canvas_h, top + h)
        if r <= l or b <= t:
            continue
        src = im.crop((l - left, t - top, r - left, b - top))
        if src.mode != "RGB":
            src = src.convert("RGB")

        bg = acc.crop((l, t, r, b))
        if mode == BlendMode.NORMAL:
            blended = src
        else:
            blended = _FAST_BLENDS[mode](bg, src)
        if not opaque:
            mask = src_a.crop((l - left, t - top, r - left, b - top))
            blended = Image.composite(blended, bg, mask)
        acc.paste(blended, (l, t))

    return acc


def render_psd(psd_file: Path) -> Image.Image:
    """Render the PSD (all layers, blend modes and opacity) to a PIL RGB image."""
    psd = PSDImage.open(str(psd_file))
    base = _fast_composite(psd)
    if base is None:
        base = psd.composite(force=True)
    if base is None:
        raise ValueError(f"Cannot render PSD: {psd_file.name}")
    # Work in RGB (drop alpha so Difference math matches Photoshop output)
    if base.mode != "RGB":
        base = base.convert("RGB")
    return base


def prepare_tif(tif_file: Path, size: tuple[int, int]) -> Image.Image:
    """Load the TIF, convert modes like the JSX did (Bitmap -> Grayscale),
    and fit it onto a canvas the same size as the PSD.

    Photoshop places the duplicated layer anchored at the top-left, so we
    paste at (0, 0). Missing areas are black (Difference with black = no-op).
    """
    tif = Image.open(str(tif_file))
    if tif.mode == "1":  # Bitmap -> Grayscale, same as the JSX script
        tif = tif.convert("L")
    tif = tif.convert("RGB")

    canvas = Image.new("RGB", size, (0, 0, 0))
    canvas.paste(tif, (0, 0))
    return canvas


def process_files(psd_file: Path, tif_file: Path, output_folder: Path) -> str:
    base = render_psd(psd_file)
    tif = prepare_tif(tif_file, base.size)

    result = ImageChops.difference(base, tif)

    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / (psd_file.stem + ".png")
    result.save(str(output_file), format="PNG")
    return f"OK  {psd_file.name} + {tif_file.name} -> {output_file.name}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge matching PSD+TIF pairs with Difference blend, export PNG."
    )
    parser.add_argument("psd_folder", nargs="?", help="Folder containing PSD files")
    parser.add_argument("tif_folder", nargs="?", help="Folder containing TIF files")
    parser.add_argument("-j", "--jobs", type=int, default=min(os.cpu_count() or 4, 10),
                        help="Number of parallel workers (default: min(CPU count, 10); "
                             "more workers are slower for very large images)")
    args = parser.parse_args()

    psd_folder = Path(args.psd_folder) if args.psd_folder else Path(input("PSD folder: ").strip().strip('"'))
    tif_folder = Path(args.tif_folder) if args.tif_folder else Path(input("TIF folder: ").strip().strip('"'))

    if not psd_folder.is_dir() or not tif_folder.is_dir():
        sys.exit("One or both folder paths are invalid. Please check the paths and try again.")

    psd_files = sorted(psd_folder.glob("*.psd"))
    tif_files = sorted(tif_folder.glob("*.tif")) + sorted(tif_folder.glob("*.tiff"))

    if not psd_files:
        sys.exit("No PSD files found in the selected PSD folder.")
    if not tif_files:
        sys.exit("No TIF/TIFF files found in the selected TIF folder.")

    tif_map = {f.stem.lower(): f for f in tif_files}
    pairs = [(p, tif_map[p.stem.lower()]) for p in psd_files if p.stem.lower() in tif_map]

    if not pairs:
        sys.exit("No matching PSD and TIF pairs were found. Make sure files share the same base name.")

    output_folder = tif_folder / "PSD_check"
    jobs = min(args.jobs, len(pairs))
    print(f"Processing {len(pairs)} pair(s) with {jobs} worker(s)...")

    processed = 0
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(process_files, p, t, output_folder): p for p, t in pairs}
        for future in as_completed(futures):
            psd_file = futures[future]
            try:
                print(future.result())
                processed += 1
            except Exception as e:  # keep going, like the JSX per-file try/catch
                print(f"Error processing {psd_file.name}: {e}", file=sys.stderr)

    if processed == 0:
        sys.exit("All pairs failed. Check the error messages above.")
    print(f"Processed {processed} PSD file(s) successfully. Output: {output_folder}")


if __name__ == "__main__":
    main()
