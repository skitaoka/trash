// cf. http://en.wikipedia.org/wiki/Slice_sampling
//   & http://www.singularpoint.org/blog/math/slice-sampling/
template <typename evaluatable_function>
std::vector<double> &&
  slice_sampling(std::size_t const num_samples,
                 double const initial_value,
                 evaluatable_function const f,
                 double const step_width,
                 std::size_t const max_slice_width)
{
  assert(step_width > 0.0);
  assert(max_slice_width > 0.0);

  double x0 = initial_value;
  double w = step_width;

  std::size_t cnt = 0;
  std::vector<double> samples(n);
  for (;;) {
    double const z = std::log(f());
    double const U = std::random();
    double L = x0 - w * U;
    double R = L + w;
    std::size_t J = std::floor(max_slice_width * std::random());
    std::size_t K = (max_slice_width - 1) - j;

    while ((J > 0) && (z < std::log(f(L)))) {
      L = L - w;
      --J;
    }
    while ((K > 0) && (z < std::log(f(R)))) {
      R = R + w;
      --K;
    }

    for (;;) {
      double const x1 = L + std::random() * (R - L);
      if (z < std::log(f(x1))) {
        samples[cnt] = x1;
        if (++cnt >= num_samples) {
          return samples;
        }
        w = (w + std::abs(x1 - x0)) * 0.5;
        x0 = x1;
        break;
      }
      ((x < x0) ? L : R) = x;
    }
  }
  return samples; // never reach here.
}
