---
layout: post
title: Graspnet on TUM-RGBD
date: 2025-12-18 10:55:00-0400
description: Implementation of Graspnet on desk2, next step is intergrate SAM3 with Graspnet
tags: research computer-vision robotics
categories: research-updates
---

## 先说目标 · Goal

**中文**：你现在已经有了完整的理论链条，接下来最关键的一步，是写一段**实际可跑的代码**，把「点云/深度图作为条件」真正接到 SAM3D 仓库的推理接口 `Inference` 上，而不是只在论文里停留在概念层面。

**English**: You already have the full theoretical chain. The next crucial step is to write **runnable code** that connects _your own depth / point cloud_ to the SAM3D `Inference` interface, so that point-map conditioning is actually used in practice rather than only in theory.

This note serves as a **bilingual engineering memo** that shows:

- how to convert a depth map + camera intrinsics into a point map;
- how to feed this external point map into `Inference` so that **MoGe is bypassed** and your own geometry is used;
- how to keep a fallback path that still uses MoGe when you do not have depth.

---

## 1. 从深度图生成 pointmap · From Depth Map to Point Map

假设你已经有：

- `rgb`: an `H×W×3` `np.uint8` RGB image
- `mask`: an `H×W` binary mask (`0/1` or `bool`)
- `depth`: an `H×W` float depth map in **meters**
- Camera intrinsics  
  \[
  K =
  \begin{bmatrix}
  fx & 0 & cx \\
  0 & fy & cy \\
  0 & 0 & 1
  \end{bmatrix}
  \]

可以用下面的函数把深度图转换到相机坐标系下的三维点图（point map）：

import numpy as np
import torch

def depth_to_pointmap(depth, K):
"""
depth: (H, W) float32, depth in meters
K: (3, 3) camera intrinsic matrix

    Returns:
        (H, W, 3) float32, point map in the camera coordinate system (x, y, z).
    """
    assert depth.ndim == 2, "depth must be (H, W)"
    H, W = depth.shape

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    ys, xs = np.meshgrid(
        np.arange(H, dtype=np.float32),
        np.arange(W, dtype=np.float32),
        indexing="ij",  # ys ~ row (y), xs ~ col (x)
    )

    zs = depth.astype(np.float32)  # (H, W)

    xs_3d = (xs - cx) * zs / fx
    ys_3d = (ys - cy) * zs / fy

    pts = np.stack([xs_3d, ys_3d, zs], axis=-1)  # (H, W, 3)
    return torch.from_numpy(pts).float()> **中文总结**：上面就是标准的 pinhole model 反投影，把每个像素的 `(u, v, depth)` 还原成相机坐标系下的 `(x, y, z)`，最终得到一个 `H×W×3` 的点图。

---

## 2. 把外部 pointmap 接到 `Inference` · Plugging Point Map into `Inference`

下面是一个最小 demo：**完全不用 MoGe，直接用你自己的深度图生成的 point map**，再把它送入 `Inference`：

import os
import numpy as np
from PIL import Image
import torch

from notebook.inference import Inference # provided by the SAM3D repo
from your_module import depth_to_pointmap # function from Section 1

def load_rgb(path):
img = Image.open(path).convert("RGB")
arr = np.array(img).astype(np.uint8) # (H, W, 3)
return arr

def load_mask(path):
m = Image.open(path)
m = np.array(m)
if m.ndim == 3: # if mask is RGBA or RGB, use the last channel
m = m[..., -1]
m = (m > 0).astype(np.uint8) # binary mask: 0/1
return m

def main(): # 1) paths & config (修改成你自己的路径)
config_file = "/home/kara/sam-3d-objects/checkpoints/your_infer_config.yaml"
rgb_path = "/home/kara/your_image.png"
mask_path = "/home/kara/your_mask.png"
depth_path = "/home/kara/your_depth.npy" # saved H×W depth map in meters

    # 2) load rgb / mask / depth
    rgb   = load_rgb(rgb_path)            # (H, W, 3), uint8
    mask  = load_mask(mask_path)          # (H, W), 0/1
    depth = np.load(depth_path).astype(np.float32)  # (H, W), meters

    # 3) camera intrinsics K  (示例值，替换成你的标定结果)
    H, W = depth.shape
    fx, fy = 1000.0, 1000.0
    cx, cy = W / 2.0, H / 2.0
    K = np.array(
        [[fx, 0.0, cx],
         [0.0, fy, cy],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    # 4) build point map from depth (H×W×3, float32, camera frame)
    pointmap = depth_to_pointmap(depth, K)   # torch.Tensor, (H, W, 3)

    # 5) call Inference with external pointmap
    infer = Inference(config_file=config_file, compile=False)

    with torch.no_grad():
        out = infer(
            image=rgb,          # (H, W, 3), uint8
            mask=mask,          # (H, W), 0/1
            seed=0,
            pointmap=pointmap,  # <<< key: Point Map Conditioning
        )

    print(out.keys())
    # e.g. out["translation"], out["scale"], out["rotation"], out["glb"], etc.

if **name** == "**main**":
main()**中文解释**：

- 如果你把 `pointmap` 作为参数传进去，`Inference.__call__` 会检测到 `pointmap is not None`：
  - 不再调用 MoGe，而是直接使用你提供的点云 / 深度几何；
  - 内部会自动做分辨率对齐、尺度归一化、从点云估计相机内参；
  - 在 `pose_decoder` 阶段，会用 `scene_scale` / `scene_center` 把预测位姿还原到真实尺度；
  - `PointPatchEmbed` 会把点云编码成条件 token，让网络“看到”每个像素背后的 3D 几何。

**English explanation**:

- When `pointmap` is provided, `Inference.__call__`:
  - **skips MoGe** and directly uses your external geometry;
  - performs resolution alignment and normalization internally;
  - uses `scene_scale` and `scene_center` in the `pose_decoder` to recover poses in real-world scale;
  - encodes the point map through `PointPatchEmbed` so that every pixel is conditioned on its underlying 3D point.

---

## 3. 只有 MoGe 时的简化用法 · Simpler Path When You Only Have MoGe

如果你当前还没有准备好深度图 / 点云，只想先把 **默认的 MoGe + Point Map Conditioning** 跑通，那么可以完全不传 `pointmap`：

infer = Inference(config_file=config_file, compile=False)

with torch.no_grad():
out = infer(
image=rgb,
mask=mask,
seed=0,
pointmap=None, # or simply omit this argument
)在这种情况下：

- MoGe 会根据 RGB + mask 先预测一个粗点云；
- 这个点云仍然会经过同样的 Point Map Conditioning 流程；
- 之后你可以逐步把自己的深度 / LiDAR 数据替换进来，验证外部点云与 MoGe 生成点云之间的效果差异。

---

## 4. 接下来可以做什么 · Next Steps

**中文**：

1. 把以上代码整理成一个独立脚本（例如 `my_pointmap_demo.py`），放到仓库的 `notebook/` 或 `tools` 目录下；
2. 把你在 TUM RGB-D（如 `freiburg1_desk2`）上的标定参数写清楚，存成 `yaml` 或 `json`；
3. 用真实的深度 + 相机内参与 RGB 图像跑通完整 pipeline，并在这篇博客下方持续记录实验结果（成功例子、失败例子、可视化截图）；
4. 最终，把这里的“工程实践笔记”提炼成论文中 Experiments / Implementation Details 的一部分。

**English**:

1. Turn the above snippets into a standalone script (e.g., `my_pointmap_demo.py`) inside your repo.
2. Store your camera intrinsics for TUM RGB-D (e.g., `freiburg1_desk2`) in a clean `yaml`/`json` file.
3. Run the full pipeline with real depth, intrinsics, and RGB, and keep logging the results (both successes and failures) in this blog post.
4. Eventually, distill these engineering notes into the _Experiments_ / _Implementation Details_ section of your paper.

---

你按这两步改完、保存，博客里就是你要的那篇中英双语 + 可跑代码的说明了。
