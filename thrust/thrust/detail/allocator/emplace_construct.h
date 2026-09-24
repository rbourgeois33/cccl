// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/detail/allocator/allocator_system.h>
#include <thrust/detail/copy.h>
#include <thrust/detail/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/zip_iterator.h>

#include <cuda/std/__host_stdlib/memory>
#include <cuda/std/__iterator/advance.h>
#include <cuda/std/__iterator/distance.h>
#include <cuda/std/__memory/allocator_traits.h>
#include <cuda/std/__memory/pointer_traits.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_trivially_copy_constructible.h>
#include <cuda/std/tuple>

THRUST_NAMESPACE_BEGIN
namespace detail
{
template <typename... Args>
struct emplace_via_allocator
{
  ::cuda::std::tuple<Args...> args_as_a_tuple;

  template <typename T>
  _CCCL_HOST_DEVICE void operator()(T& p)
  {
    // apply allows to have any numer of args as input to function
    ::cuda::std::apply(
      [&](auto&... xs) {
        new (static_cast<void*>(&p)) T(xs...);
      },
      args_as_a_tuple);
  }
};

// Build one object at p from args, on the system (CPU or GPU) the allocator belongs to
template <typename Allocator, typename Pointer, typename... Args>
_CCCL_HOST_DEVICE void emplace_construct(Allocator& a, Pointer ptr, Args... args)
{
  // Dispatch according to backend (loop/kernel)
  thrust::for_each_n(allocator_system<Allocator>::get(a), ptr, 1, emplace_via_allocator<Args...>{{args...}});
}
} // namespace detail
THRUST_NAMESPACE_END
