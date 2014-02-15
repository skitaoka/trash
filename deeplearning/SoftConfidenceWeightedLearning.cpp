#include "OnlineLearning.hpp"

int main(int argc, char* argv[])
{
  std::size_t const epoch = 1;
  std::size_t const num_samples = 100;

  // 教師データ
  std::vector<std::vector<double>> x(num_samples);
  std::vector<int> y(num_samples); // [-1, 1]

  double const phi  = 1.0; // パラメータ 1: φ
  double const C    = 1.0; // パラメータ 2
  double const phi2 = aka::square(phi);
  double const psi  = 1.0 + phi2 / 2;
  double const zeta = 1.0 + phi2;
  std::vector<double> mu;    // 平均 μ
  std::vector<double> Sigma; // 分散 Σ
  for (std::size_t n = 0; n < epoch; ++n) {
    for (std::size_t t = 0, size = x.size(); t < size; ++t) {
      double const yT = y[t];
      double const mT = yT * aka::inner_product(mu, x[t]); // margin
      double const vT = aka::quadratic_form(Sigma, x[t]);  // confidnece

#if 0
      // SCW-1:
      //   argmin_{μ,Σ} D_KL(N(μ,Σ) || N(μ[t],Σ[t])) + C l^{φ}(N(μ,Σ); (x[t], y[t])),
      //     where l^{φ}(N(μ,Σ); (x[t], y[t])) = max{0, φ sqrt(x[t]^T Σ x[t]) - y[t] μ^T x[t]}
      double const alphaT = std::min(C, (std::sqrt(aka::square(0.5 * mT * phi2) + vT * phi2 * zeta) - mT * psi) / (vT * zeta));
#else
      // SCW-2:
      //   argmin_{μ,Σ} D_KL(N(μ,Σ) || N(μ[t],Σ[t])) + C l^{φ}(N(μ,Σ); (x[t], y[t]))^2,
      //     where l^{φ}(N(μ,Σ); (x[t], y[t])) = max{0, φ sqrt(x[t]^T Σ x[t]) - y[t] μ^T x[t]}
      double const nT = vT + 0.5 / C;
      double const gammaT = phi * std::sqrt(aka::square(phi * mT * vT) + 4.0 * nT * vT * (nT + vT * phi2));
      double const alphaT = 0.5 * (gammaT - (2.0 * nT + phi2 * vT) * mT) / (nT * (nT + vT * phi2));
#endif
      if (alphaT < 0.0) {
        continue;
      }

      double const wT = alphaT * vT * phi;
      double const zT = 0.5 * wT;
      double const uT = aka::square(std::sqrt(aka::square(zT) + vT) - zT);
      double const betaT = alphaT * phi / (std::sqrt(uT) + wT);

      // update mean
      double const sT = alphaT * yT;
      aka::transform(mu, Sigma, x[t],
        [sT](double const muT, double const SigmaT, double const xT)
      {
        double const rT = xT * SigmaT;
        return muT + sT * rT;
      });

      // update variance:
      aka::transform(Sigma, x[t],
        [betaT](double const SigmaT, double const xT)
      {
        double const rT = xT * SigmaT;
        return SigmaT - betaT * aka::square(rT);
      });
    }
  }

  return 0;
}
