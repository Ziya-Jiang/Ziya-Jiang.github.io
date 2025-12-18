---
layout: post
title: Graspnet on TUM-RGBD
date: 2025-12-18 10:55:00-0400
description: Implementation of Graspnet on desk2, next step is intergrate SAM3 with Graspnet
tags: research computer-vision robotics
categories: research-updates
---

# demo.py 详细实现文档

## 概述

`demo.py` 是 GraspNet 基线模型的演示脚本，用于从 RGB-D 图像中检测抓取姿态。该脚本实现了完整的抓取检测流程：从图像加载、点云生成、模型推理、碰撞检测到可视化。

### Overview

`demo.py` is a demonstration script for the GraspNet baseline model, designed to detect grasp poses from RGB-D images. The script implements a complete grasp detection pipeline: from image loading, point cloud generation, model inference, collision detection to visualization.

The pipeline processes RGB-D input to generate 3D point clouds, uses a deep learning model to predict potential grasp configurations, filters out collisions with the scene, and visualizes the final grasp candidates. This implementation serves as a foundation for integrating advanced segmentation models like SAM3 in future work.

{% include figure.liquid
    path="assets/img/teaser.png"
    title="GraspNet Pipeline Overview"
    caption="Figure 1. The complete GraspNet pipeline: from RGB-D input to grasp pose prediction. The system processes point clouds through feature extraction, approach direction prediction, clustering, and parallel prediction of grasp parameters and quality scores."
    class="img-fluid rounded z-depth-1"
%}

{% include figure.liquid
    path="assets/img/desk2_grasp.png"
    title="Grasp Detection Results on Desk2 Scene"
    caption="Figure 2. Visualization of detected grasp poses on the desk2 scene. The point cloud shows various objects (books, boxes, electronic devices) with colored line segments representing geometric features. The green gripper models indicate the top-K predicted grasp configurations after collision filtering."
    class="img-fluid rounded z-depth-1"
%}

## 目录结构

1. [导入和初始化](#导入和初始化)
2. [命令行参数解析](#命令行参数解析)
3. [模型加载函数](#模型加载函数)
4. [数据处理函数](#数据处理函数)
5. [抓取检测函数](#抓取检测函数)
6. [碰撞检测函数](#碰撞检测函数)
7. [可视化函数](#可视化函数)
8. [主函数](#主函数)

---

## 导入和初始化

### 核心库导入

```python
import os, sys, numpy as np
import open3d as o3d  # 点云处理和可视化
import argparse, yaml
from PIL import Image
import torch
```

### 项目模块导入

- `graspnet`: GraspNet 模型和预测解码函数
- `graspnet_dataset`: 数据集类（未直接使用，但导入以保持兼容性）
- `collision_detector`: 无模型碰撞检测器
- `data_utils`: 相机信息和点云生成工具

### 路径设置

```python
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
```

---

## 命令行参数解析

### 参数列表

| 参数                 | 类型  | 默认值   | 说明                 |
| -------------------- | ----- | -------- | -------------------- |
| `--checkpoint_path`  | str   | **必需** | 模型检查点文件路径   |
| `--num_point`        | int   | 20000    | 点云采样点数         |
| `--num_view`         | int   | 300      | 抓取视角数量         |
| `--collision_thresh` | float | 0.01     | 碰撞检测阈值         |
| `--voxel_size`       | float | 0.01     | 碰撞检测时的体素大小 |

---

## 模型加载函数

### `get_net()`

**功能**: 初始化并加载训练好的 GraspNet 模型

**实现细节**:

1. **模型初始化**

   ```python
   net = GraspNet(
       input_feature_dim=0,      # 不使用额外特征（仅使用坐标）
       num_view=cfgs.num_view,   # 300个视角
       num_angle=12,             # 12个角度离散化
       num_depth=4,              # 4个深度离散化
       cylinder_radius=0.05,     # 圆柱体半径
       hmin=-0.02,               # 最小高度
       hmax_list=[0.01,0.02,0.03,0.04],  # 最大高度列表
       is_training=False         # 推理模式
   )
   ```

2. **设备选择**

   - 优先使用 CUDA (GPU)
   - 回退到 CPU

3. **检查点加载**

   - 从指定路径加载 PyTorch 检查点
   - 提取 `model_state_dict` 和 `epoch` 信息
   - 加载模型权重到网络

4. **模型设置**
   - 设置为评估模式 (`net.eval()`)
   - 禁用 dropout 和 batch normalization 的训练行为

**返回**: 配置好的 GraspNet 模型实例

---

## 数据处理函数

### `get_and_process_data(data_dir)`

**功能**: 从 RGB-D 图像生成点云并预处理

**输入**: `data_dir` - 数据目录路径

**输出**:

- `end_points`: 包含点云数据的字典
- `cloud`: Open3D 点云对象（用于可视化）

### 详细流程

#### 1. 图像加载 (行 54-56)

```python
color_img = Image.open(os.path.join(data_dir, 'desk2.png'))
color = np.array(color_img, dtype=np.float32) / 255.0
depth = np.array(Image.open(os.path.join(data_dir, 'desk2_d.png')))
```

- 加载 RGB 图像和深度图像
- 将 RGB 归一化到 [0, 1] 范围

#### 2. 深度图预处理 (行 58-60)

```python
if len(depth.shape) == 3:
    depth = depth[:, :, 0]  # 多通道时取第一通道
```

- 确保深度图为单通道

#### 3. 图像尺寸匹配 (行 62-70)

```python
if color.shape[:2] != depth.shape[:2]:
    # 使用 LANCZOS 重采样调整 RGB 图像尺寸以匹配深度图
    color_img = color_img.resize((depth.shape[1], depth.shape[0]), Image.Resampling.LANCZOS)
    color = np.array(color_img, dtype=np.float32) / 255.0
```

- 确保 RGB 和深度图尺寸一致
- 使用高质量重采样算法

#### 4. 相机内参加载 (行 72-86)

从 `freiburg1_desk2.yaml` 文件读取相机参数：

```yaml
camera_params:
  image_height: 480
  image_width: 640
  fx: 517.3
  fy: 516.5
  cx: 318.6
  cy: 255.3
  png_depth_scale: 5000.0
```

- `fx, fy`: 焦距（像素单位）
- `cx, cy`: 主点坐标
- `png_depth_scale`: 深度缩放因子（将深度值转换为米）

#### 5. 点云生成 (行 88-90)

```python
camera = CameraInfo(width, height, fx, fy, cx, cy, factor_depth)
cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
```

- 使用相机内参和深度图生成 3D 点云
- `organized=True`: 保持点云的图像形状 (H, W, 3)

**点云生成原理**:

```
z = depth / scale
x = (u - cx) * z / fx
y = (v - cy) * z / fy
```

其中 `(u, v)` 是像素坐标。

#### 6. 有效点提取 (行 92-99)

```python
mask = (depth > 0)  # 只保留有效深度点
cloud_masked = cloud[mask]
color_masked = color[mask]  # 或处理灰度图
```

- 过滤无效深度值（深度为 0 的点）
- 同步提取对应的颜色信息

#### 7. 点云采样 (行 101-109)

```python
if len(cloud_masked) >= cfgs.num_point:
    # 随机采样指定数量的点
    idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
else:
    # 如果点数不足，先取所有点，再重复采样补足
    idxs1 = np.arange(len(cloud_masked))
    idxs2 = np.random.choice(len(cloud_masked),
                            cfgs.num_point-len(cloud_masked),
                            replace=True)
    idxs = np.concatenate([idxs1, idxs2], axis=0)
```

- 将点云采样到固定数量（默认 20000 点）
- 确保模型输入尺寸一致

#### 8. 数据转换 (行 111-120)

```python
# Open3D 点云对象（用于可视化）
cloud = o3d.geometry.PointCloud()
cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))

# PyTorch 张量（用于模型推理）
cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
cloud_sampled = cloud_sampled.to(device)
end_points['point_clouds'] = cloud_sampled  # (1, N, 3)
end_points['cloud_colors'] = color_sampled   # (N, 3)
```

- 创建两个版本的点云：
  - Open3D 对象用于可视化
  - PyTorch 张量用于模型推理

---

## 抓取检测函数

### `get_grasps(net, end_points)`

**功能**: 使用模型进行前向传播并解码抓取预测

**实现细节**:

1. **前向传播** (行 126-128)

   ```python
   with torch.no_grad():  # 禁用梯度计算以节省内存
       end_points = net(end_points)
       grasp_preds = pred_decode(end_points)
   ```

2. **预测解码** (`pred_decode` 函数)

   该函数将模型的原始输出转换为可用的抓取姿态：

   **输入** (`end_points` 包含):

   - `objectness_score`: 物体性分数 (B, Ns, 2)
   - `grasp_score_pred`: 抓取分数 (B, Ns, A, D)
   - `fp2_xyz`: 种子点坐标 (B, Ns, 3)
   - `grasp_top_view_xyz`: 最佳视角方向 (B, Ns, 3)
   - `grasp_angle_cls_pred`: 角度分类预测 (B, Ns, A)
   - `grasp_width_pred`: 抓取宽度预测 (B, Ns, A, D)
   - `grasp_tolerance_pred`: 容差预测 (B, Ns, A, D)

   **处理步骤**:

   a. **角度解码**:

   ```python
   grasp_angle_class = torch.argmax(grasp_angle_class_score, 0)
   grasp_angle = grasp_angle_class.float() / 12 * np.pi
   ```

   - 从 12 个离散角度中选择最佳角度
   - 转换为弧度值

   b. **深度解码**:

   ```python
   grasp_depth_class = torch.argmax(grasp_score, 1, keepdims=True)
   grasp_depth = (grasp_depth_class.float()+1) * 0.01
   ```

   - 从 4 个离散深度中选择最佳深度
   - 深度范围: 0.01m - 0.04m

   c. **物体性过滤**:

   ```python
   objectness_pred = torch.argmax(objectness_score, 0)
   objectness_mask = (objectness_pred==1)
   ```

   - 只保留被识别为物体的点

   d. **旋转矩阵生成**:

   ```python
   rotation_matrix = batch_viewpoint_params_to_matrix(approaching_, grasp_angle_)
   ```

   - 从视角方向和角度生成 3x3 旋转矩阵

   e. **最终输出格式**:

   ```python
   [score, width, height, depth, rotation_matrix(9), center(3), obj_id]
   ```

   - 每个抓取姿态包含 17 个参数

3. **转换为 GraspGroup** (行 129-130)
   ```python
   gg_array = grasp_preds[0].detach().cpu().numpy()
   gg = GraspGroup(gg_array)
   ```
   - 将 PyTorch 张量转换为 NumPy 数组
   - 创建 `GraspGroup` 对象用于后续处理

**返回**: `GraspGroup` 对象，包含所有检测到的抓取姿态

---

## 碰撞检测函数

### `collision_detection(gg, cloud)`

**功能**: 检测抓取姿态与场景点云的碰撞

**实现细节**:

1. **初始化检测器** (行 134)

   ```python
   mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
   ```

   - 使用无模型碰撞检测器（不需要物体模型）
   - `voxel_size`: 用于点云下采样的体素大小（默认 0.01m）

2. **碰撞检测** (行 135)

   ```python
   collision_mask = mfcdetector.detect(
       gg,
       approach_dist=0.05,           # 接近距离
       collision_thresh=cfgs.collision_thresh  # 碰撞阈值
   )
   ```

   **检测原理**:

   - 将场景点云转换到每个抓取姿态的坐标系
   - 检查抓取器（夹爪）内部是否有场景点
   - 计算碰撞 IoU（交并比）
   - 如果 IoU > `collision_thresh`，标记为碰撞

   **抓取器模型**:

   - 宽度: `grasp_width`
   - 高度: 0.02m
   - 深度: `grasp_depth`
   - 手指宽度: 0.01m
   - 手指长度: 0.06m

3. **过滤碰撞抓取** (行 136)
   ```python
   gg = gg[~collision_mask]
   ```
   - 移除所有发生碰撞的抓取姿态

**返回**: 过滤后的 `GraspGroup` 对象

---

## 可视化函数

### `vis_grasps(gg, cloud)`

**功能**: 可视化点云和抓取姿态

**实现细节**:

1. **非极大值抑制 (NMS)** (行 140)

   ```python
   gg.nms()
   ```

   - 移除空间上过于接近的重复抓取

2. **按分数排序** (行 141)

   ```python
   gg.sort_by_score()
   ```

   - 按抓取质量分数降序排列

3. **选择 Top-K** (行 142)

   ```python
   gg = gg[:50]
   ```

   - 只保留前 50 个最佳抓取

4. **转换为可视化格式** (行 143)

   ```python
   grippers = gg.to_open3d_geometry_list()
   ```

   - 将抓取姿态转换为 Open3D 几何对象（夹爪模型）

5. **显示** (行 144)
   ```python
   o3d.visualization.draw_geometries([cloud, *grippers])
   ```
   - 使用 Open3D 可视化器显示点云和抓取器

---

## 主函数

### `demo(data_dir)`

**功能**: 执行完整的抓取检测流程

**执行流程**:

```python
def demo(data_dir):
    # 1. 加载模型
    net = get_net()

    # 2. 处理数据
    end_points, cloud = get_and_process_data(data_dir)

    # 3. 检测抓取
    gg = get_grasps(net, end_points)

    # 4. 碰撞检测（如果启用）
    if cfgs.collision_thresh > 0:
        gg = collision_detection(gg, np.array(cloud.points))

    # 5. 可视化
    vis_grasps(gg, cloud)
```

### 主程序入口

```python
if __name__=='__main__':
    data_dir = 'doc/example_data'
    demo(data_dir)
```

---

## 数据流图

```
RGB图像 (desk2.png)
    ↓
深度图像 (desk2_d.png)
    ↓
相机内参 (freiburg1_desk2.yaml)
    ↓
    点云生成
    ↓
    点云采样 (20000点)
    ↓
    ┌─────────────────┐
    │  GraspNet 模型  │
    └─────────────────┘
    ↓
    原始预测输出
    ↓
    预测解码 (pred_decode)
    ↓
    抓取姿态 (GraspGroup)
    ↓
    碰撞检测 (可选)
    ↓
    可视化
```

---

## 关键参数说明

### 模型参数

- **num_view (300)**: 抓取视角的离散化数量
- **num_angle (12)**: 抓取角度的离散化数量（每个视角 12 个角度）
- **num_depth (4)**: 抓取深度的离散化数量（4 个深度级别）
- **cylinder_radius (0.05m)**: 用于特征提取的圆柱体半径

### 数据处理参数

- **num_point (20000)**: 输入模型的点云数量
- **png_depth_scale (5000.0)**: TUM 数据集的深度缩放因子

### 碰撞检测参数

- **collision_thresh (0.01)**: 碰撞检测阈值
- **voxel_size (0.01m)**: 点云下采样的体素大小
- **approach_dist (0.05m)**: 抓取接近距离

---

## 输出格式

### GraspGroup 结构

每个抓取姿态包含以下信息：

| 索引  | 名称     | 说明           | 单位 |
| ----- | -------- | -------------- | ---- |
| 0     | score    | 抓取质量分数   | -    |
| 1     | width    | 抓取宽度       | m    |
| 2     | height   | 抓取高度       | m    |
| 3     | depth    | 抓取深度       | m    |
| 4-12  | rotation | 旋转矩阵 (3x3) | -    |
| 13-15 | center   | 抓取中心坐标   | m    |
| 16    | obj_id   | 物体ID         | -    |

---

## 注意事项

1. **内存管理**:

   - 使用 `torch.no_grad()` 禁用梯度计算
   - 点云采样限制内存使用

2. **设备兼容性**:

   - 自动检测 CUDA 可用性
   - 支持 CPU 和 GPU 推理

3. **图像格式**:

   - 支持 RGB 和灰度图像
   - 自动处理尺寸不匹配

4. **深度图格式**:

   - 支持单通道和多通道深度图
   - 深度值需要根据数据集缩放

5. **可视化**:
   - 需要图形界面支持
   - 在服务器上运行时可能需要 X11 转发

---

## 扩展建议

1. **批量处理**: 修改以支持批量图像处理
2. **结果保存**: 添加抓取结果保存功能
3. **性能优化**: 使用 TensorRT 或 ONNX 加速推理
4. **多线程**: 并行处理多个图像
5. **Web 界面**: 添加 Web 可视化界面

---

## 相关文件

- `models/graspnet.py`: GraspNet 模型定义
- `models/modules.py`: 模型子模块
- `utils/data_utils.py`: 数据处理工具
- `utils/collision_detector.py`: 碰撞检测实现
- `graspnetAPI/`: GraspNet API 和评估工具

---
