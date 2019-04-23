/*
 * @Author: Jake Gu
 * @Date: 2019-04-21 15:28:35
 * @LastEditTime: 2019-04-21 15:30:13
 */
#ifndef __TYPE_H__
#define __TYPE_H__

#include <cstdint>
#include <string>
#include "utils/Namespace.h"
#include "utils/Assert.h"

PROJECT_NAMESPACE_BEGIN

// Built-in type aliases
using IndexType = std::uint32_t;
using IntType = std::int32_t;
using RealType = double;
using Byte = std::uint8_t;

using FlowIntType = std::int64_t;

// Built-in type constants
constexpr IndexType INDEX_TYPE_MAX = 1000000000; // 1e+9
constexpr IntType INT_TYPE_MAX = 1000000000;     // 1e+9
constexpr IntType INT_TYPE_MIN = -1000000000;    // -1e+9
constexpr RealType REAL_TYPE_MAX = 1e100;
constexpr RealType REAL_TYPE_MIN = -1e100;
constexpr RealType REAL_TYPE_TOL = 1e-6;

PROJECT_NAMESPACE_END

#endif // __TYPE_H__
