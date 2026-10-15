#pragma once

#include "image.hpp"

enum class OverFlowPolicy
{
    CLAMP,
    WRAP,
};

enum class CenterPointBox
{
    Corners = 0,
    Center = 1,
};

struct BoxCandidate {
    float score;
    int index;
};

