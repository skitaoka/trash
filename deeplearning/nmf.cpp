//
// Algorithm implementation of Non-Negative Matrix Factorization.
//
//   cf. http://d.hatena.ne.jp/a_bicky/20100325/1269479839
//
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>
 
// measures NMF error.
template <std::size_t M, std::size_t N>
void nmf_error(double Y[M][N], double GF[M][N])
{
  double err = 0.0;

  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t i = 0; i < N; ++i) {
      assert(GF[j][i] >= 1.0e-12);
      err += Y[j][i] * std::log(Y[j][i] / GF[j][i]) - Y[j][i] + GF[j][i];
    }
  }

  printf("NMF err = %f\n", err);
}
 
//
// Given a non-negative matrix Y(n x m),
// find non-negative matrix factors G(n x r) and F(r x m) such that:
//  Y = G F.
//
//   F_{ij} <- F_{ij} \sum_{k} G_{ki} ( Y_{kj} / (GF)_{kj} )
//   G_{ij} <- G_{ij} \sum_{k} ( Y_{ik} / (GF)_{kj} ) F_{jk}
//   G_{ij} <- G_{ij} / \sum_{k} G_{kj}
//
template <std::size_t M, std::size_t N, std::size_t R>
void nmf(double const Y[M][N], double G[R][M], double F[M][R], std::size_t const maxloop = 10)
{
  double GF[M][N]; // approximation of the target matrix
 
  // initialize G and F with random seeds
  for (std::size_t j = 0; j < R; j++) {
    for (std::size_t i = 0; i < M; i++) {
      G[j][i] = randomMT();
    }
  }
  for (std::size_t j = 0; j < M; j++) {
    for (std::size_t i = 0; i < R; i++) {
      F[j][i] = randomMT();
    }
  }

  GF = G * F;

  for (std::size_t loop = 0; loop < maxloop; ++loop) {
    std::printf("iteration %d of %d\n", loop, maxloop);

    std::puts("  updating Fij...");
    for (std::size_t j = 0; j < M; ++j) {
      for (std::size_t i = 0; i < R; ++i) {
        double sum = 0.0;
        for (std::size_t k = 0; k < N; ++k) {
          assert(GF[j][j] >= 1.0e-12);
          sum += G[i][k] * (Y[j][k] / GF[j][k]);
        }
        F[j][i] *= sum;
      }
    }
 
    std::puts("  updating Gij 1...");
    for (std::size_t j = 0; j < r; ++j) {
      for (std::size_t i = 0; i < n; ++i) {
        double sum = 0.0;
        for (std::size_t k = 0; k < m; ++k) {
          assert(GF[k][i] >= 1.0e-12);
          sum += (Y[k][i] / GF[k][i]) * F[k][j];
        }
        G[j][i] *= sum;
      }
    }
 
    std::puts("updating Gij 2...");
    for (std::size_t j = 0; j < r; ++j) {
      double sum = 0.0;
      for (std::size_t k = 0; k < n; k++) {
        sum += G[j][k];
      } 
      assert(sum >= 1.0e-12);

      sum = 1.0 / sum;
      for (std::size_t i = 0; i < n; i++) {
        G[j][i] *= sum;
      }
    }
 
    GF = G * F;

    nmf_error(Y, GF, n, m);
  }
}
