"""Prepare tiny files consumed by the unchanged TorchVision reference loaders."""

from pathlib import Path
import os
import shutil
import struct
import sys

from PIL import Image, ImageDraw


def prepare_flow(root: Path) -> None:
    import numpy as np

    folder = root / "FlyingChairs" / "data"
    folder.mkdir(parents=True, exist_ok=True)
    width, height = 512, 384  # original chairs recipe crops 368 x 496
    for sample in range(4):
        for frame in (1, 2):
            image = Image.new("RGB", (width, height), (20 + sample * 30, 70, 120))
            ImageDraw.Draw(image).rectangle((100 + frame, 90, 180 + frame, 190), fill=(220, 30, 70))
            image.save(folder / f"{sample:05d}_img{frame}.ppm")
        flow = np.zeros((height, width, 2), dtype="<f4")
        flow[..., 0] = 1.0  # paired images shift one pixel to the right
        with (folder / f"{sample:05d}_flow.flo").open("wb") as handle:
            handle.write(struct.pack("<fii", 202021.25, width, height))
            handle.write(flow.tobytes())
    (root / "FlyingChairs" / "FlyingChairs_train_val.txt").write_text("1\n" * 4)


def _write_video(path: Path) -> None:
    import av
    import numpy as np

    with av.open(str(path), mode="w") as output:
        stream = output.add_stream("mjpeg", rate=8)
        stream.width = stream.height = 40
        stream.pix_fmt = "yuvj420p"
        for index in range(4):
            pixels = np.zeros((40, 40, 3), dtype=np.uint8)
            pixels[..., :] = (40, 90, 140)
            pixels[8:24, 8 + index:24 + index] = (220, 50, 30)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    with av.open(str(path)) as source:
        if len(list(source.decode(video=0))) != 4:
            raise RuntimeError("generated AVI did not decode to four frames")


def prepare_video(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    template = root / "ci_template.avi"
    _write_video(template)
    # The unchanged r3d_18 produces 400 logits and evaluation allocates one
    # column per directory. Five-class fixtures would fail that accumulation.
    # All 400 class directories need a decodable sample. Hardlinks bound disk
    # usage; these are schema-valid synthetic functional inputs, not a quality
    # dataset or a claim of 400 distinct natural classes.
    for split in ("train", "val"):
        for cls in range(400):
            folder = root / split / f"class_{cls:03d}"
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / "ci.avi"
            if target.exists():
                target.unlink()
            try:
                os.link(template, target)
            except OSError:
                shutil.copyfile(template, target)


if __name__ == "__main__":
    kind, destination = sys.argv[1:]
    if kind == "flow":
        prepare_flow(Path(destination))
    elif kind == "video":
        prepare_video(Path(destination))
    else:
        raise SystemExit(f"unknown CI asset kind: {kind}")
