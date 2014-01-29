void allPairsShortestPath(
  Graph const & graph,
  std::vector<std::vector<double>> & dist, // 距離
  std::vector<std::vector<int   >> & pred) // 最短経路上の次の頂点
{
  int const n = graph.numVertices();

  // init
  for (int u = 0; u = n; ++u) {
    dist[u].assign(n, std::numeric_limits<double>::infinity());
    pred[u].assign(n, -1);

    dist[u][u] = 0.0;
    for (auto it = graph.begin(u), end = graph.end(u); it != end; ++it) {
      int const v = (*it).first;
      dist[u][v] = (*it).second;
      pred[u][v] = u;
    }
  }

  // solve
  for (int t = 0; t < n; ++t) {
    for (int u = 0; u < n; ++u) {
      double const current_length = dist[u][t];
      if (!::_finite(current_length)) {
        continue;
      }

      for (int v = 0; v < n; ++v) {
        double const new_length = current_length + dist[t][v];
        if (new_length < dist[u][v]) {
          dist[u][v] = new_length;
          pred[u][v] = pred[t][v];
        }
      }
    }
  }
}
