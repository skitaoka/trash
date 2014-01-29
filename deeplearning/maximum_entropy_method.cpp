#include <cstdlib>
#include <memory>
#include <limits>
#include <algorithm>

std::size_t const N = 100; // データ数
std::size_t const D = 100; // 特徴ベクトルの次元数
std::size_t const L = 10 ; // 分類するカテゴリ数
double const threshold = std::log(0.99999);

double      x   [N][D];       // DxN 行列: 訓練データ
std::size_t y   [N];          // N 次元ベクトル: 正解カテゴリ -> {0,L-1}
double      w   [L][D];       // LxD 行列: 重み行列
double      umax[N];          // 
double      logz[N];          // 
double      buf [N][L];
double      (&u)[N][L] = buf; // LxN 行列: 生のエネルギー: U = W.X
double      (&l)[N][L] = buf; // LxN 行列
double      (&p)[N][L] = buf; // LxN 行列
double      g   [L][D];       // LxD 行列: 勾配

void learn()
{
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    // 生のエネルギーを計算: U = W.X
#ifdef _OPENMP
#pragma omp for
#endif
    for (std::size_t j = 0; j < N; ++j) {
      for (std::size_t i = 0; i < L; ++i) {
        double retval = 0.0;
        for (std::size_t k = 0; k < D; ++k) {
          retval += w[i][k] * x[j][k];
        }
        u[j][i] = retval;
      }
    }

    // インスタンスごとの最大のエネルギーを求める
    //   umax[j] = max_i u[j][i]
#ifdef _OPENMP
#pragma omp for
#endif
    for (std::size_t j = 0; j < N; ++j) {
      double retval = -std::numeric_limits<double>::infinity();
      for (std::size_t i = 0; i < L; ++i) {
        retval = std::max(retval, u[j][i]);
      }
      umax[j] = retval;
    }

    // 対数正規化
    //   logZ = log { ∑_i exp(u[j][i]) }
    //        = log { ∑_i exp(u[j][i] - umax[j]) * exp(umax[j]) }
    //        = log { ∑_i exp(u[j][i] - umax[j]) } + log { exp(umax[j]) }
    //        = log { ∑_i exp(u[j][i] - umax[j]) } + umax[j]
#ifdef _OPENMP
#pragma omp for
#endif
    for (std::size_t j = 0; j < N; ++j) {
      double retval = 0.0;
      for (std::size_t i = 0; i < L; ++i) {
        retval += std::exp(u[j][i] - umax[j]);
      }
      logz[j] = umax[j] + std::log(retval);
    }

    // 正規化対数確率
#ifdef _OPENMP
#pragma omp for
#endif
    for (std::size_t j = 0; j < N; ++j) {
      for (std::size_t i = 0; i < L; ++i) {
        l[j][i] = u[j][i] - logz[j];
      }
    }

    // 対数確率の和 (最大化する値)
    double v = 0.0;
#ifdef _OPENMP
#pragma omp for reduction(+:v)
#endif
    for (std::size_t j = 0; j < N; ++j) {
      v += l[j][y[j]];
    }

    if (v > threshold) {
      // 収束した
    }

    // 確率: P
#ifdef _OPENMP
#pragma omp for
#endif
    for (std::size_t j = 0; j < N; ++j) {
      for (std::size_t i = 0; i < L; ++i) {
        p[j][i] = std::exp(l[j][i]);
      }
    }

    // 勾配: G = P X^T
#ifdef _OPENMP
#pragma omp for
#endif
    for (std::size_t i = 0; i < L; ++i) {
      for (std::size_t k = 0; k < D; ++k) {
        double retval = 0.0;
        for (std::size_t j = 0; j < N; ++j) {
          retval -= p[j][i] * x[j][k];
        }
        g[i][k] = retval;
      }
    }
    for (std::size_t j = 0; j < N; ++j) {
      for (std::size_t k = 0; k < D; ++k) {
        g[y[j]][k] += x[j][k];
      }
    }
  }
}
