// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace viz
{

// Display backend for a VizSession. In its own header so VizSession
// and VizCompositor can both reference it without an include cycle.
enum class DisplayMode
{
    kOffscreen,
    kWindow,
    kXr,
};

} // namespace viz
