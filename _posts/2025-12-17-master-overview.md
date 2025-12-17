---
layout: post
title: Master Academic Journey Overview
date: 2025-12-17 23:50:00-0400
description: A comprehensive summary of my threqe-week summer research progress,including concept map integration, system demonstrations, and future plans
tags: research computer-vision robotics
categories: research-updates
---

{% include figure.liquid
    path="assets/img/robot.png"
    title="Master Thesis Overview"
    caption="图 1. 本图展示了硕士阶段研究工作的整体结构，包括研究背景、核心方法（如 Gaussian Memory Field 与 3D Gaussian Splatting）、主要应用场景以及论文之间的关系。自下而上依次表示基础理论与工具、系统方法设计和上层应用/项目，实现从方法到真实机器人系统落地的完整闭环。"
    class="img-fluid rounded z-depth-1"
%}

# Architecture Overview: Semantic-Geometric Memory for Embodied AI

This figure illustrates the proposed framework for an Embodied AI system capable of understanding open-ended human instructions, constructing a hierarchical memory representation, and performing physically robust interactions. The pipeline consists of five interconnected stages:

## 1. Multimodal Perception & Feature Extraction (Top)

The system takes **RGB** and **Depth** streams as input.

- **Visual Odometry (VO):** Estimates the initial camera pose to establish a spatial reference frame.
- **Semantic Extraction:** Utilizes a suite of state-of-the-art vision models—**CLIP** (Contrastive Language-Image Pre-training), **SAM** (Segment Anything Model), and **DINO**—to extract dense, semantically enhanced multimodal features from key frames.

## 2. The Bridge Module & Memory Bank (Center)

The **Bridge Module** acts as a filter and distributor, processing key frames and their associated poses to construct a comprehensive **Memory Bank**. This memory is hierarchically organized into three distinct layers:

- **Visual Memory:** Stores semantic features correlated with spatial positions, enabling the system to "know" what objects look like and what they are.
- **Spatial Memory:** Generated via a **Large Language Model (LLM)**, this layer abstracts high-level spatial relationships between objects (e.g., "the cup is _on_ the table," "the chair is _next to_ the desk").
- **Geometry Memory:** Utilizes a **Feed-Forward Network (FFN)** to reconstruct object-level geometric meshes and textures. It incorporates scale information to perform **scale correction**, ensuring the digital reconstruction matches physical reality.

## 3. Interaction Pose Regression (Left Pillar)

This module is responsible for planning _how_ to approach an object.

- Based on the interaction object category, it retrieves the specific target from the **Memory Bank**.
- It synthesizes data from Visual, Spatial, and Geometry memories to calculate and screen for the **optimal interaction viewpoint**, ensuring the robot approaches the object from a viable angle.

## 4. Geometry-Based Force Constraint (Right Pillar)

This module ensures the physical feasibility and stability of the grasp.

- It retrieves the detailed surface geometry (including **surface normals**) from the Geometry Memory.
- It performs real-time **hand localization** relative to the object's surface.
- It applies **Force Closure** optimization algorithms to refine the grasping pose, ensuring the grasp is mechanically stable and prevents object slippage.

## 5. VLM Inference & Execution (Bottom)

The system is driven by high-level intent through **VLM (Vision-Language Model) Inference**.

- **Instruction Understanding:** The VLM processes open-ended **Human Instructions**.
- **Reasoning:** It reasons over the Memory Field to identify one or multiple target objects and infers their best interaction poses.
- **Execution:**
  - **Perception:** Uses the Gaussian memory field for real-time relocalization.
  - **Action:** Outputs low-level control commands for **Navigation** (moving to the target) and **Grasping** (executing the optimized interaction).
