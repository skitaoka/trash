// cf. Markov Chain Monte Carlo Method without Detailed Balance; 
//     http://d.hatena.ne.jp/teramonagi/20111120/1321785155
double w[N]; // 分布
double c[N]; // c[i] = c[i-1] + w[i], where c[-1] = 0;

int metropolis(int const current)
{
  int const candidate = random(0, N); // mutation
  double const w_candidate = w[candidate];
  double const w_current   = w[current];
  return ((w_candidate >= w_current) || (random() * w_current < w_candidate)) ? candidate : current;
}

int suwa_todo(int const current)
{
  double const xi = random() * w[current];

  double sum = 0.0;
  for (int candidate = 0; candidate < N; ++candidate) {
    sum += v(current, candidate);
    if (xi < sum) {
      return candidate
    }
  }
}

double v(int const i, int const j)
{
  double const w_i = w[i]; // w_{i}
  double const w_j = w[j]; // w_{j}
  double const s_i = c[i]; // S_{i}
  double const s_jm1 = (j == 0) ? c[N] : c[j-1]; // S_{j-1}
  double const delta_ij = s_i - s_jm1 + w[0];
  return std::max(0.0,
    std::min(
      std::min(delta_ij, w_i + w_j - delta_ij),
      std::min(w_i, w_j)));
}
