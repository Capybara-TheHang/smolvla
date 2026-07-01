// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <viz/core/viz_types.hpp>
#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

namespace viz
{

// Per-layer rects from tile_layout(): outer is the equal-slice tile
// (used as scissor); content is the aspect-fit rect inside outer
// (used as viewport). Margins between them keep the clear color —
// free letterbox.
struct TileSlot
{
    VkRect2D outer{};
    VkRect2D content{};
};

// Row-major aspect-preserving grid. cols = ceil(sqrt(N)), rows =
// ceil(N / cols). padding is the inter-tile gap in pixels. Empty
// input -> empty output.
std::vector<TileSlot> tile_layout(const std::vector<float>& aspects, Resolution fb_size, uint32_t padding = 0);

} // namespace viz
