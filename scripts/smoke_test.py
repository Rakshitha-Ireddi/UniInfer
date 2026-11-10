"""Smoke test: one image question under a few configs. Shows the answer, time and peak VRAM.

    CUDA_VISIBLE_DEVICES=0 python scripts/smoke_test.py
"""
import time

import torch
from PIL import Image, ImageDraw

from uniinfer.runtime import QwenVLRunner


def make_image():
    img = Image.new("RGB", (640, 480), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([40, 40, 260, 200], fill="red")
    d.ellipse([360, 60, 560, 260], fill="blue")
    d.text((60, 330), "INVOICE TOTAL: $482.17", fill="black")
    return img


def main():
    r = QwenVLRunner()
    img = make_image()
    q = "What colors are the two shapes, and what is the invoice total written in the image?"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    for v_ratio, kv_ratio in [(1.0, 1.0), (0.5, 1.0), (0.5, 0.5), (0.25, 0.25)]:
        r.run(img, q, v_ratio, "dissim", kv_ratio, 8)  # warm-up
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize(); t = time.perf_counter()
        ans = r.run(img, q, v_ratio=v_ratio, v_method="dissim", kv_ratio=kv_ratio, max_new_tokens=48)
        torch.cuda.synchronize()
        print(f"visual {v_ratio:4.0%}  kv {kv_ratio:4.0%}  {time.perf_counter() - t:5.2f}s  "
              f"peak {torch.cuda.max_memory_allocated() / 1e9:4.2f} GB  ->  {ans}")


if __name__ == "__main__":
    main()
