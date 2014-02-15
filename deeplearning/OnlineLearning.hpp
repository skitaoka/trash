#pragma once

#ifndef AKA_ONLINELEARNING_HPP_INCLUDED
#define AKA_ONLINELEARNING_HPP_INCLUDED

#include <algorithm>
#include <vector>
#include <numeric>

namespace aka
{
  template <typename T>
  inline T square(T const x)
  {
    return x * x;
  }

  template <typename T>
  inline T cube(T const x)
  {
    return x * aka::square(x);
  }

  template <std::size_t N, typename T>
  inline T pow(T const x)
  {
    switch (N) {
    case 0: return 1;
    case 1: return x;
    case 2: return aka::square(x);
    default:
      if (N & 1) {
        return aka::square(aka::pow<N / 2>(x)) * x;
      }
      else {
        return aka::square(aka::pow<N / 2>(x));
      }
    }
  }

  template <typename T>
  inline T clamp(T const value, T const _min, T const _max)
  {
    return std::min(std::max(value, _min), _max);
  }

  template <typename T>
  inline T inner_product(std::vector<T> const & a, std::vector<T> const & b)
  {
    return std::inner_product(a.begin(), a.end(), b.begin(), T());
  }

  template <typename T>
  inline T quadratic_form(std::vector<T> const & a, std::vector<T> const & x)
  {
    return std::inner_product(a.begin(), a.end(), x.begin(), T(), std::plus<T>(),
      [](T const a, T const x) { return a * x * x; });
  }

  template <typename T, typename Fn>
  inline void transform(std::vector<T> & a, std::vector<T> const & b, Fn fn)
  {
    for (std::size_t i = 0, size = a.size(); i < size; ++i) {
      a[i] = fn(a[i], b[i]);
    }
  }

  template <typename T, typename Fn>
  inline void transform(std::vector<T> & a, std::vector<T> const & b, std::vector<T> const & c, Fn fn)
  {
    for (std::size_t i = 0, size = a.size(); i < size; ++i) {
      a[i] = fn(a[i], b[i], c[i]);
    }
  }
}

#endif//AKA_ONLINELEARNING_HPP_INCLUDED
