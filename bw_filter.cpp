/**
 * cf. https://docs.google.com/document/d/1w-jQlG3yfEjiWNzSgZIHV_gm2FvHI7IKsELSLVuP6ow/edit
 */

/**
 * bi-quad filter
 */
class biquad_filter
{
public:
  biquad_filter()
    : x1_(0.0), x2_(0.0)
    , y1_(0.0), y2_(0.0)
  {
  }

  /**
   * Butter-worth Low-Pass Filter
   * omega_c: Cutoff frequency [0, Pi]
   * s: sin(omega_c);
   * c: cos(omega_c);
   */
  void set_bw_low_pass(int const k, int const n, double const s, double const c)
  {
    double const sk = s * std::sin(M_PI * (2.0 * k + 1.0) / (2.0 * n));
    double const a0 = 1.0 / (1.0 + sk); // this is a rcp of a0.
    {
      double const b1 = (+1.0 - c) * a0;
      b0_ = b1 * +0.5;
      b1_ = b1;
      // where b2 = b0.
    }
    {
      a1_ =  c  * 2.0  * a0;
      a2_ = (sk - 1.0) * a0;
    }
  }

  /**
   * Butter-worth High-Pass Filter
   * omega_c: Cutoff frequency [0, Pi]
   * s: sin(omega_c);
   * c: cos(omega_c);
   */
  void set_bw_high_pass(int const k, int const n, double const s, double const c)
  {
    double const sk = s * std::sin(M_PI * (2.0 * k + 1.0) / (2.0 * n));
    double const a0 = 1.0 / (1.0 + sk); // this is a rcp of a0.
    {
      double const b1 = (-1.0 - c) * a0;
      b0_ = b1 * -0.5;
      b1_ = b1;
      // where b2 = b0.
    }
    {
      a1_ =  c  * 2.0  * a0;
      a2_ = (sk - 1.0) * a0;
    }
  }


public:
  inline double operator () (double const x0)
  {
    double const y0
      = b0_ * (x0 + x2_)
      + b1_ * x1_
      + a1_ * y1_
      + a2_ * y2_;
    x2_ = x1_; x1_ = x0;
    y2_ = y1_; y1_ = y0;
    return y0;
  }
 
private:
  double b0_;
  double b1_;
  double a1_;
  double a2_;

  double x1_;
  double x2_;
  double y1_;
  double y2_;
};

class biquad_filter_x
{
public:
  biquad_filter()
    : x1_(_mm_setzero_pd())
    , x2_(_mm_setzero_pd())
    , y1_(_mm_setzero_pd())
    , y2_(_mm_setzero_pd())
  {
  }

  /**
   * Butter-worth Low-Pass Filter
   * omega_c: Cutoff frequency [0, Pi]
   * s: sin(omega_c);
   * c: cos(omega_c);
   */
  void set_bw_low_pass(int const k, int const n, double const s, double const c)
  {
    double const sk = s * std::sin(M_PI * (2.0 * k + 1.0) / (2.0 * n));
    double const sk = s * std::sin(M_PI * (2.0 * k + 1.0) / (2.0 * n));
    double const a0 = 1.0 / (1.0 + sk); // this is a rcp of a0.
    {
      double const b1 = (+1.0 - c) * a0;
      b0_ = _mm_set1_pd(b1 * +0.5);
      b1_ = _mm_set1_pd(b1);
      // where b2 = b0.
    }
    {
      a1_ = _mm_set1_pd( c  * 2.0  * a0);
      a2_ = _mm_set1_pd((sk - 1.0) * a0);
    }
  }

  /**
   * Butter-worth High-Pass Filter
   * omega_c: Cutoff frequency [0, Pi]
   * s: sin(omega_c);
   * c: cos(omega_c);
   */
  void set_bw_high_pass(int const k, int const n, double const s, double const c)
  {
    double const sk = s * std::sin(M_PI * (2.0 * k + 1.0) / (2.0 * n));
    double const a0 = 1.0 / (1.0 + sk); // this is a rcp of a0.
    {
      double const b1 = (-1.0 - c) * a0;
      b0_ = _mm_set1_pd(b1 * -0.5);
      b1_ = _mm_set1_pd(b1);
      // where b2 = b0.
    }
    {
      a1_ = _mm_set1_pd( c  * 2.0  * a0);
      a2_ = _mm_set1_pd((sk - 1.0) * a0);
    }
  }


public:
  inline __m128d operator () (__m128d const x0)
  {
    __m128d const y0
      = _mm_add_pd(
          _mm_add_pd(
            _mm_mul_pd(b0_, _mm_add_pd(x0, x2_))
            _mm_mul_pd(b1_, x1_)),
          _mm_add_pd(
            _mm_mul_pd(a1_, y1_),
            _mm_mul_pd(a2_, y2_)));
    x2_ = x1_; x1_ = x0;
    y2_ = y1_; y1_ = y0;
    return y0;
  }
 
private:
  __m128d b0_;
  __m128d b1_;
  __m128d a1_;
  __m128d a2_;

  __m128d x1_;
  __m128d x2_;
  __m128d y1_;
  __m128d y2_;
};

/**
 * Butter-worth Low-Pass Filter
 * (N has to be a odd number.)
 */
template <int N>
class bw_low_pass_filter
{
public:
  /**
   * fs: Sampling Frequency
   * we assume a transformation to 48kHz.
   */
  void set(int const fs)
  {
    assert((fs == 96000) || (fs == 192000) || (fs == 384000));
    double const omega_c = (M_PI * 2.0 * 20000.0) / fs;
    double const s = std::sin(omega_c);
    double const c = std::cos(omega_c);
    for (int k = 0； k < N/2; ++k) {
      bf[k].set_bw_low_pass(k, N, s, c);
    }
  }

public:
  inline double operator () (double const x) {
    double y = x;
    for (int k = 0; k < N/2; ++k) {
      y = bf[k](y);
    }
    return y;
  }

private:
  biquad_filter bf[N/2];
};

template <int N>
class bw_low_pass_filter_x
{
public:
  /**
   * fs: Sampling Frequency
   * we assume a transformation to 48kHz.
   */
  void set(int const fs)
  {
    assert((fs == 96000) || (fs == 192000) || (fs == 384000));
    double const omega_c = (M_PI * 2.0 * 20000.0) / fs;
    double const s = std::sin(omega_c);
    double const c = std::cos(omega_c);
    for (int k = 0； k < N/2; ++k) {
      bf[k].set_bw_low_pass(k, N, s, c);
    }
  }

public:
  inline __m128d operator () (__m128d const x) {
    __m128d y = x;
    for (int k = 0; k < N/2; ++k) {
      y = bf[k](y);
    }
    return y;
  }

private:
  biquad_filter_x bf[N/2];
};
