// ƒMƒuƒ“ƒX‰ñ“]‚É‚æ‚é QR •ª‰ð

inline void make_rot(double const x, double const y, double * const c, double * const s)
{
  double const r = std::sqrt(x * x + y * y);
  *c = x / r;
  *s = y / r;
}

inline void rot(double * const x, double * const y, double const c, double const s)
{
  double const u = *x;
  double const v = *y;
  *x = c * u + s * v;
  *y = c * v - s * u;
}

void solve(matrix_t & a, vector_t & v)
{
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t k = i + 1; k < n; ++k) {
      double c;
      double s;
      make_rot(a(i, i), a(k, i), &c, &s);
      for (std::size_t j = i + 1; j < n; ++j) {
        rot(&a(i, j), &a(k, j), c, s);
      }
    }
  }

  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i + 1; j < n; ++j) {
      b[i] -= a(i, j) * b[j];
    }
    b[i] /= a(i, i);
  }
}
