// D 次元のベクトルから L ビットのハッシュ値を生成する.
// cf. http://d.hatena.ne.jp/tsubosaka/20090928/1254147181
std::vector<bool> make_lsh(std::vector<std::pair<std::size_t, double>> const & x, std::size_t const L)
{
  std::vector<double> v(L);
  for (std::size_t i = 0, size = x.size(); i < size; ++i) {
    std::vector<bool> const hash = make_hash(x[i].first); // L bit のハッシュ値    
    double const value = x[i].second;
    for (std::size_t j = 0; j < L; ++j) {
      v[j] += hash[j] ? value : -value;
    }
  }

  std::vector<bool> lsh(L);
  for (std::size_t j = 0; j < L; ++j) {
    lsh[j] = (v[j] > 0.0);
  }
  return lsh;
}
