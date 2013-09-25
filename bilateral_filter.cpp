// Bilateral Filter
for (int s = 0; s < n; ++s) {
  double weight = 0.0;
  double signal = 0.0;
  for (int p = 0; p < n; ++p) {
    double const w = f(p - s) * g(I[p] - I[s]);
    weight += w;
    signal += w * I[p];
  }
  J[s] = signal / weight;
}
