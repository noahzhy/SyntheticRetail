"""Visualize bounding boxes on an image.

Expected label JSON format (list):
[
  {"label": "Can.006", "bbox": [xmin, ymin, xmax, ymax], ...},
  ...
]

The bbox can be either:
- normalized floats in [0, 1] (auto-detected), or
- absolute pixel coordinates.
"""

from __future__ import annotations

import argparse
import colorsys
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class BBoxItem:
	label: str
	xmin: float
	ymin: float
	xmax: float
	ymax: float
	occluded: bool = False


def _load_bbox_items(labels_path: str) -> list[BBoxItem]:
	with open(labels_path, "r", encoding="utf-8") as f:
		payload = json.load(f)

	if not isinstance(payload, list):
		raise ValueError(
			f"Unsupported labels JSON. Expected a list of items, got {type(payload).__name__}."
		)

	items: list[BBoxItem] = []
	for idx, obj in enumerate(payload):
		if not isinstance(obj, dict):
			raise ValueError(f"Item #{idx} must be an object/dict, got {type(obj).__name__}.")
		label = str(obj.get("label", f"item_{idx}"))
		bbox = obj.get("bbox")
		if (
			not isinstance(bbox, (list, tuple))
			or len(bbox) != 4
			or not all(isinstance(v, (int, float)) for v in bbox)
		):
			raise ValueError(
				f"Item #{idx} has invalid 'bbox'. Expected [xmin,ymin,xmax,ymax] numbers, got: {bbox!r}"
			)
		xmin, ymin, xmax, ymax = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
		occluded = bool(obj.get("occluded", False))
		items.append(
			BBoxItem(label=label, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, occluded=occluded)
		)

	return items


def _is_normalized(items: Iterable[BBoxItem]) -> bool:
	vals: list[float] = []
	for it in items:
		vals.extend([it.xmin, it.ymin, it.xmax, it.ymax])
	if not vals:
		return False
	vmin = min(vals)
	vmax = max(vals)
	return vmin >= -1e-6 and vmax <= 1.0 + 1e-6


def _clamp(v: float, lo: float, hi: float) -> float:
	return max(lo, min(hi, v))


def _color_for_label(label: str) -> tuple[int, int, int]:
	# Deterministic vivid color from label (high saturation + high value).
	h = 2166136261
	for ch in label:
		h ^= ord(ch)
		h = (h * 16777619) & 0xFFFFFFFF

	# Hue in [0,1). Keep S/V high for strong visibility.
	hue = ((h % 360) / 360.0)
	sat = 0.95
	val = 1.0
	rf, gf, bf = colorsys.hsv_to_rgb(hue, sat, val)
	return int(rf * 255), int(gf * 255), int(bf * 255)


def _to_pixel_bbox(it: BBoxItem, w: int, h: int, normalized: bool) -> tuple[int, int, int, int]:
	if normalized:
		xmin = it.xmin * w
		xmax = it.xmax * w
		ymin = it.ymin * h
		ymax = it.ymax * h
	else:
		xmin, ymin, xmax, ymax = it.xmin, it.ymin, it.xmax, it.ymax

	# Clamp + sanitize.
	xmin = _clamp(xmin, 0.0, float(w - 1))
	xmax = _clamp(xmax, 0.0, float(w - 1))
	ymin = _clamp(ymin, 0.0, float(h - 1))
	ymax = _clamp(ymax, 0.0, float(h - 1))

	# Ensure correct ordering.
	x1 = int(math.floor(min(xmin, xmax)))
	y1 = int(math.floor(min(ymin, ymax)))
	x2 = int(math.ceil(max(xmin, xmax)))
	y2 = int(math.ceil(max(ymin, ymax)))
	return x1, y1, x2, y2


def draw_bboxes(
	image_path: str,
	items: list[BBoxItem],
	*,
	normalized: bool | None,
	thickness: int,
	font_size: int,
	hide_occluded: bool = False,
) -> "Any":
	try:
		from PIL import Image, ImageDraw, ImageFont  # type: ignore
	except Exception as e:  # pragma: no cover
		raise RuntimeError(
			"Missing dependency: Pillow. Install with: pip install pillow"
		) from e

	img = Image.open(image_path).convert("RGB")
	w, h = img.size

	use_normalized = _is_normalized(items) if normalized is None else normalized
	draw = ImageDraw.Draw(img)

	# Prefer a scalable TTF font (bigger labels), fall back to PIL's built-in.
	font = None
	try:
		font = ImageFont.truetype("DejaVuSans.ttf", size=max(8, int(font_size)))
	except Exception:  # pragma: no cover
		try:
			font = ImageFont.load_default()
		except Exception:  # pragma: no cover
			font = None

	for it in items:
		if hide_occluded and it.occluded:
			continue
		x1, y1, x2, y2 = _to_pixel_bbox(it, w, h, use_normalized)
		color = (170, 170, 170) if it.occluded else _color_for_label(it.label)

		# BBox rectangle.
		t = max(1, int(thickness))
		draw.rectangle([x1, y1, x2, y2], outline=color, width=t)

		# Label background + text.
		text = it.label
		if font is not None:
			l, t0, r, b = draw.textbbox((0, 0), text, font=font)
			tw, th = (r - l), (b - t0)
		else:  # pragma: no cover
			tw, th = draw.textlength(text), 12

		pad = 4
		tx1 = x1
		ty1 = max(0, y1 - (th + 2 * pad))
		tx2 = min(w - 1, x1 + tw + 2 * pad)
		ty2 = min(h - 1, ty1 + th + 2 * pad)
		draw.rectangle([tx1, ty1, tx2, ty2], fill=color)
		draw.text((tx1 + pad, ty1 + pad), text, fill=(0, 0, 0), font=font)

	return img


def _build_argparser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description="Visualize [xmin,ymin,xmax,ymax] bboxes on an image")
	p.add_argument("--labels", required=True, help="Path to labels JSON")
	p.add_argument("--image", required=True, help="Path to image")
	p.add_argument("--save", default=None, help="Output image path (e.g. preview.png)")
	p.add_argument("--show", action="store_true", help="Show preview window")
	p.add_argument("--no-show", action="store_true", help="Do not show preview window")
	p.add_argument(
		"--normalized",
		action="store_true",
		help="Force treat bbox as normalized [0,1] coordinates",
	)
	p.add_argument(
		"--absolute",
		action="store_true",
		help="Force treat bbox as absolute pixel coordinates",
	)
	p.add_argument("--hide-occluded", action="store_true", help="Do not draw occluded bounding boxes")
	p.add_argument("--thickness", type=int, default=4, help="BBox line thickness")
	p.add_argument("--font-size", type=int, default=20, help="Label font size")
	return p


def main(argv: list[str] | None = None) -> int:
	args = _build_argparser().parse_args(argv)

	if args.normalized and args.absolute:
		raise SystemExit("Use only one of --normalized or --absolute")

	if not os.path.exists(args.labels):
		raise SystemExit(f"labels not found: {args.labels}")
	if not os.path.exists(args.image):
		raise SystemExit(f"image not found: {args.image}")

	items = _load_bbox_items(args.labels)
	normalized: bool | None
	if args.normalized:
		normalized = True
	elif args.absolute:
		normalized = False
	else:
		normalized = None

	img = draw_bboxes(
		args.image,
		items,
		normalized=normalized,
		thickness=args.thickness,
		hide_occluded=args.hide_occluded,
		font_size=args.font_size,
	)

	if args.save:
		img.save(args.save)
		print(f"saved: {args.save}")

	# Default behavior: show unless explicitly disabled.
	want_show = args.show or (not args.no_show and args.save is None)
	if want_show:
		img.show()

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
