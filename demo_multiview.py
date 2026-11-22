"""Command-line demo: one textured 3D model from an unposed set of 1-4 images of the same object.

Run from the repository root on a Linux CUDA machine with the upstream dependencies, extensions and weights:

    python demo_multiview.py --images a.png b.png c.png --output-dir outputs/run1

Stages: background removal (only for images without a usable alpha channel), geometry with
``Hunyuan3DUnposedMultiViewPipeline`` (``shape.glb``), texture with ``Hunyuan3DPaintPipeline`` using every image as
a reference (``textured.obj`` plus its maps), and packing into a PBR GLB (``model.glb``). See ``docs/usage.md``.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import os
import struct
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent
# Same package search path as demo.py, made absolute.
sys.path.insert(0, str(REPO_ROOT / "hy3dshape"))
sys.path.insert(0, str(REPO_ROOT / "hy3dpaint"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

MODEL_PATH = "tencent/Hunyuan3D-2.1"
REALESRGAN_CKPT = REPO_ROOT / "hy3dpaint" / "ckpt" / "RealESRGAN_x4plus.pth"
MAX_SEED = 2**32 - 1  # The texture stage seeds NumPy, which accepts 0 .. 2**32 - 1.
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

SHAPE_OUTPUTS = ("shape.glb",)
# Files written by Hunyuan3DPaintPipeline (remesh, OBJ with maps) and by the GLB packing step.
TEXTURE_OUTPUTS = (
    "white_mesh_remesh.obj",
    "textured.obj",
    "textured.mtl",
    "textured.jpg",
    "textured_metallic.jpg",
    "textured_roughness.jpg",
    "model.glb",
)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    from hy3dshape.multiview_pipeline import FUSION_MODES, MAX_IMAGES, normalize_image_set

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--images", nargs="+", required=True, metavar="PATH",
        help=f"1 to {MAX_IMAGES} images of the same object, in any order and from any viewpoints.",
    )
    parser.add_argument("--fusion", choices=FUSION_MODES, default="velocity_mean", help="Geometry fusion strategy.")
    parser.add_argument("--seed", type=int, default=0, help=f"Seed for geometry and texture, 0 to {MAX_SEED}.")
    parser.add_argument(
        "--output-dir", required=True, type=Path,
        help="Directory for this run's outputs. It may exist but must not contain outputs of an earlier run.",
    )
    parser.add_argument("--shape-only", action="store_true", help="Stop after the geometry stage (shape.glb).")
    args = parser.parse_args(argv)

    # Reuse the pipeline's own input check so that too many images fail here, before background removal and
    # weight loading. Unreadable files are reported by _load_images().
    try:
        normalize_image_set(args.images)
    except ValueError as exc:
        parser.error(str(exc))
    if not 0 <= args.seed <= MAX_SEED:
        parser.error(f"--seed must be between 0 and {MAX_SEED}, got {args.seed}.")
    if not args.shape_only and not REALESRGAN_CKPT.is_file():
        parser.error(
            f"texture stage needs {REALESRGAN_CKPT}; download RealESRGAN_x4plus.pth there (see docs/usage.md) "
            f"or pass --shape-only."
        )

    args.output_dir = args.output_dir.resolve()
    names = SHAPE_OUTPUTS if args.shape_only else SHAPE_OUTPUTS + TEXTURE_OUTPUTS
    existing = [str(args.output_dir / name) for name in names if (args.output_dir / name).exists()]
    if existing:
        parser.error(f"outputs of an earlier run already exist: {', '.join(existing)}. Use a new --output-dir.")
    return args


def _png_colour_key(path: str) -> Optional[Tuple[int, int, Tuple[int, ...]]]:
    """``(bit_depth, colour_type, key)`` of the tRNS colour key in a PNG file, or ``None`` if it has none.

    Only grayscale (colour type 0, one sample) and truecolour (colour type 2, three samples) PNGs have a colour
    key. A palette tRNS chunk holds one alpha value per palette entry instead, and Pillow reads that correctly.
    The file has already been decoded by Pillow, so its chunk structure is valid.
    """
    with open(path, "rb") as stream:
        if stream.read(8) != _PNG_SIGNATURE:
            return None
        bit_depth = colour_type = -1
        while True:
            head = stream.read(8)
            if len(head) < 8:
                return None
            length, kind = struct.unpack(">I4s", head)
            if kind == b"IHDR":
                header = stream.read(length)  # width and height (8 bytes), then bit depth and colour type
                bit_depth, colour_type = header[8], header[9]
            elif kind == b"tRNS":
                data = stream.read(length)
                if colour_type == 0 and length == 2:
                    return bit_depth, colour_type, struct.unpack(">H", data)
                if colour_type == 2 and length == 6:
                    return bit_depth, colour_type, struct.unpack(">3H", data)
                return None
            elif kind in (b"IDAT", b"IEND"):
                return None  # tRNS has to come before the image data
            else:
                stream.seek(length, os.SEEK_CUR)
            stream.seek(4, os.SEEK_CUR)  # CRC


def _colour_key_to_alpha(image: Image.Image, path: str) -> Image.Image:
    """Turn the tRNS colour key of a PNG into an alpha channel; any other image is returned unchanged.

    A pixel is transparent exactly when all of its samples equal the key at the file's bit depth. Pillow's own
    ``convert("RGBA")`` does not give that mask for every file (checked with Pillow 10.4.0, 11.3.0 and 12.3.0): 2-
    and 4-bit gray pixels are expanded to 8 bits while the key is not scaled, the 1-bit key is scaled only by
    12.3.0, 16-bit gray pixels are clipped to 255, and 16-bit RGB samples lose their low byte when the file is read.
    Lossless PNG optimisers write all of these from ordinary RGBA cut-outs. So the key is read from the file and
    compared with the full samples here, and only afterwards are 16-bit colours reduced to 8 bits.

    Returns an RGBA image whose alpha is 0 or 255.
    """
    found = _png_colour_key(path)
    if found is None:
        return image
    bit_depth, colour_type, key = found
    if bit_depth < 8:
        # Grayscale only. Pillow expands these pixels to 8 bits by the scaling below, which maps different samples
        # to different values, so comparing after scaling the key the same way is still exact.
        samples = np.asarray(image.convert("L"))  # (H, W) uint8; 1-bit files are read as mode 1
        key = (key[0] * 255 // (2**bit_depth - 1),)
    elif bit_depth == 16 and colour_type == 2:
        # OpenCV keeps all 16 bits. It returns BGR, with an alpha channel appended when the file has a colour key.
        # imdecode() on the file's bytes because imread() cannot open non-ASCII paths on Windows.
        decoded = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise ValueError(
                f"Could not read image {path}: OpenCV failed to decode this 16-bit PNG, which is needed to apply "
                f"its transparent colour. Save it again as an 8-bit PNG or as a PNG with an alpha channel."
            )
        samples = decoded[..., 2::-1]  # (H, W, 3) uint16, RGB
    else:
        # 8-bit files as Pillow read them, or 16-bit grayscale, which Pillow reads with its full samples (I;16).
        samples = np.asarray(image)  # (H, W) or (H, W, 3)
    height, width = samples.shape[:2]
    opaque = (samples.reshape(height, width, -1) != key).any(axis=-1)  # (H, W) bool
    if bit_depth == 16:
        samples = (samples >> 8).astype(np.uint8)  # the high byte, as Pillow keeps for 16-bit RGB and RGBA
    rgba = Image.fromarray(samples).convert("RGBA")
    rgba.putalpha(Image.fromarray(opaque.astype(np.uint8) * 255))
    return rgba


def _alpha_extrema(image: Image.Image) -> Optional[Tuple[int, int]]:
    """``(min, max)`` of the image's alpha, or ``None`` if it carries no transparency information.

    Transparency is either an alpha channel or a transparency entry in ``image.info``, as Pillow keeps for palette
    images; ``convert("RGBA")`` turns that entry into an alpha channel. PNG colour keys are not left to that
    conversion: _colour_key_to_alpha() has already replaced them by an alpha channel.
    """
    if image.mode not in ("RGBA", "LA", "PA") and "transparency" not in image.info:
        return None
    return image.convert("RGBA").getchannel("A").getextrema()


def _load_images(paths: List[str]) -> List[Image.Image]:
    """Open each image and give it an alpha foreground mask; returns RGBA PIL images in input order."""
    from hy3dshape.rembg import BackgroundRemover

    remover = None
    images = []
    for path in paths:
        try:
            with Image.open(path) as opened:
                image = opened.copy()
        except OSError as exc:
            raise ValueError(f"Could not read image {path}: {exc}") from exc

        # A PNG colour key becomes an alpha channel first, so the three-way split below and the RGBA image that is
        # kept both use that one mask.
        image = _colour_key_to_alpha(image, path)
        if image.mode in ("I;16", "I;16B"):
            # 16-bit grayscale without transparency, from a PNG or a TIFF. Pillow clips these pixels to 255 when it
            # converts them, which would hand background removal a nearly white picture. Keep the high byte, as
            # Pillow itself does for 16-bit RGB.
            reduced = Image.fromarray((np.asarray(image) >> 8).astype(np.uint8))  # (H, W) uint8, mode L
            # Keep the metadata: background removal applies the EXIF orientation, as for every other photo.
            reduced.info = dict(image.info)
            image = reduced
        # Look at the original mode first: demo.py converts to RGBA before checking for RGB, so it never removes the
        # background.
        alpha = _alpha_extrema(image)
        if alpha is not None and alpha[1] == 0:
            # Dropping the alpha channel here would hand the hidden RGB values to background removal as a photo.
            raise ValueError(
                f"Image {path} is fully transparent, so it shows no object. Pass an image whose alpha channel "
                f"marks the object, or one without an alpha channel."
            )
        if alpha is not None and alpha[0] < 255:
            # Keep an existing cut-out.
            image = image.convert("RGBA")
        else:
            if remover is None:
                remover = BackgroundRemover()
            image = remover(image.convert("RGB"))

        bbox = image.getchannel("A").getbbox()
        # ImageProcessorV2.recenter() rejects a foreground that is a single row or column.
        if bbox is None or bbox[2] - bbox[0] < 2 or bbox[3] - bbox[1] < 2:
            raise ValueError(
                f"No usable foreground in {path} after background removal. Use a photo where the object is clearly "
                f"separated from the background, or pass a PNG with an alpha mask."
            )
        images.append(image)

    del remover
    gc.collect()
    return images


def _peak() -> str:
    if not torch.cuda.is_available():
        return "peak allocated n/a"
    gib = torch.cuda.max_memory_allocated() / 2**30
    return f"peak allocated {gib:.2f} GiB (PyTorch allocator peak, not whole-GPU usage)"


@contextlib.contextmanager
def _stage(name: str) -> Iterator[float]:
    """Time one stage and track its allocator peak, both starting before the stage loads its models.

    Yields the start time. A failure inside the stage (for example CUDA out of memory while loading) is reported
    with the stage name and re-raised unchanged.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        yield start
    except Exception:
        # Only to name the failed stage; the error itself is re-raised. No synchronize() here: after a CUDA error
        # it can raise again and hide the original failure.
        print(f"[demo_multiview] {name} FAILED after {time.perf_counter() - start:.1f} s, {_peak()}", flush=True)
        raise
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(
        f"[demo_multiview] {name}, model loading + inference: {time.perf_counter() - start:.1f} s, {_peak()}",
        flush=True,
    )


def _generate_shape(images: List[Image.Image], fusion: str, seed: int, shape_path: Path) -> None:
    from hy3dshape.multiview_pipeline import Hunyuan3DUnposedMultiViewPipeline

    name = f"geometry ({len(images)} image(s), fusion={fusion if len(images) > 1 else 'single'})"
    with _stage(name) as start:
        pipeline = Hunyuan3DUnposedMultiViewPipeline.from_pretrained(MODEL_PATH)
        print(f"[demo_multiview] geometry models loaded after {time.perf_counter() - start:.1f} s", flush=True)
        meshes = pipeline(images, fusion=fusion, generator=torch.manual_seed(seed))

    mesh = meshes[0]
    if mesh is None or len(mesh.faces) == 0:
        raise RuntimeError(
            "Geometry generation returned an empty mesh. Check that every input shows the same object with a clean "
            "foreground, or try another --seed."
        )
    mesh.export(str(shape_path))
    print(f"[demo_multiview] wrote {shape_path} ({len(mesh.vertices)} vertices, {len(mesh.faces)} faces)", flush=True)

    # Free the geometry models before the texture models are loaded; both do not fit on one GPU at once.
    del pipeline, meshes, mesh
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _generate_texture(images: List[Image.Image], seed: int, output_dir: Path) -> Path:
    from torchvision_fix import apply_fix

    # Must run before the super-resolution dependencies (basicsr) are imported.
    apply_fix()
    from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline

    with _stage(f"texture ({len(images)} reference image(s))") as start:
        config = Hunyuan3DPaintConfig(max_num_view=6, resolution=512)
        config.realesrgan_ckpt_path = str(REALESRGAN_CKPT)
        config.multiview_cfg_path = str(REPO_ROOT / "hy3dpaint" / "cfgs" / "hunyuan-paint-pbr.yaml")
        config.custom_pipeline = str(REPO_ROOT / "hy3dpaint" / "hunyuanpaintpbr")
        pipeline = Hunyuan3DPaintPipeline(config)
        print(f"[demo_multiview] texture models loaded after {time.perf_counter() - start:.1f} s", flush=True)
        obj_path = pipeline(
            mesh_path=str(output_dir / "shape.glb"),
            image_path=images,
            output_mesh_path=str(output_dir / "textured.obj"),
            save_glb=False,
            seed=seed,
        )
    print(f"[demo_multiview] wrote {obj_path}", flush=True)
    return Path(obj_path)


def _pack_glb(obj_path: Path, glb_path: Path) -> None:
    """Pack the textured OBJ and its albedo / metallic / roughness maps into one PBR GLB."""
    import pygltflib
    from hy3dpaint.convert_utils import create_glb_with_pbr_materials

    base = obj_path.with_suffix("")
    albedo = Path(f"{base}.jpg")
    metallic = Path(f"{base}_metallic.jpg")
    roughness = Path(f"{base}_roughness.jpg")
    for path in (albedo, metallic, roughness):
        if not path.is_file():
            raise RuntimeError(f"Texture stage did not write {path}; check the texture stage log above.")

    # create_glb_with_pbr_materials() writes temp.glb and mr_combined.png into the current directory and labels
    # every embedded image as image/png, so run it in a private directory and hand it a PNG albedo.
    with tempfile.TemporaryDirectory(prefix="glb-", dir=str(glb_path.parent)) as work_dir:
        albedo_png = Path(work_dir) / "albedo.png"
        with Image.open(albedo) as opened:
            opened.save(albedo_png)
        textures = {"albedo": str(albedo_png), "metallic": str(metallic), "roughness": str(roughness)}
        previous_dir = os.getcwd()
        os.chdir(work_dir)
        try:
            create_glb_with_pbr_materials(str(obj_path), textures, str(glb_path))
        finally:
            os.chdir(previous_dir)

    gltf = pygltflib.GLTF2().load(str(glb_path))
    primitives = [primitive for mesh in gltf.meshes for primitive in mesh.primitives]
    if not primitives or not gltf.images:
        raise RuntimeError(f"{glb_path} has no mesh or no embedded textures; GLB packing failed.")
    for primitive in primitives:
        material = None if primitive.material is None else gltf.materials[primitive.material]
        pbr = None if material is None else material.pbrMetallicRoughness
        if pbr is None or pbr.baseColorTexture is None or pbr.metallicRoughnessTexture is None:
            raise RuntimeError(
                f"{glb_path} has a primitive without base-color and metallic-roughness textures; GLB packing failed."
            )
    print(f"[demo_multiview] wrote {glb_path}", flush=True)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    images = _load_images(args.images)
    print(f"[demo_multiview] loaded {len(images)} image(s) in {time.perf_counter() - start:.1f} s", flush=True)

    _generate_shape(images, args.fusion, args.seed, args.output_dir / "shape.glb")
    if args.shape_only:
        return

    obj_path = _generate_texture(images, args.seed, args.output_dir)
    _pack_glb(obj_path, args.output_dir / "model.glb")


if __name__ == "__main__":
    main()
