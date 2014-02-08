#include <iostream>
#include <vector>
#include <limits>
#include <queue>
#include <sstream>
#include <iterator>

typedef std::pair<unsigned, unsigned> unsigned_pair;

std::vector<unsigned>
  shortest_path(
    std::vector<std::vector<unsigned_pair>> const & adj,
    unsigned const s, unsigned const t)
{
  // (s からの最短距離, 最短経路上の 1 つ前のノード)
  std::vector<unsigned_pair>
    path(adj.size(),
         std::make_pair(
           std::numeric_limits<unsigned>::max(),
           std::numeric_limits<unsigned>::max()));
  path[s] = std::make_pair(0U, s);

  std::priority_queue<unsigned_pair, std::vector<unsigned_pair>, std::less<unsigned_pair>> queue;
  queue.push(std::make_pair(0U, s));
  while (!queue.empty()) {
    auto const node = queue.top(); queue.pop();
    for (std::size_t i = 0, size = adj[node.second].size(); i < size; ++i) {
      auto     const a = adj[node.second][i];
      unsigned const d = node.first + a.second;
      if (path[a.first].first > d) {
        path[a.first] = std::make_pair(d, node.second);
        queue.push(std::make_pair(d, a.first));
      }
    }
  }

#ifndef NDEBUG
  std::transform(path.begin(), path.end(),
    std::ostream_iterator<std::string>(std::cout, "\n"),
    [](unsigned_pair const & p) {
      std::ostringstream out;
      out << "(" << p.first << "," << p.second << ")";
      return out.str();
    });
#endif

  std::vector<unsigned> retval;
  retval.push_back(t);
  while (retval.back() != s) {
    unsigned const v = retval.back();
    if (v == std::numeric_limits<unsigned>::max()) {
      return std::vector<unsigned>();
    }
    retval.push_back(path[v].second);
  }
  std::vector<unsigned>(retval.rbegin(), retval.rend()).swap(retval);
  return retval;
}

int main()
{
  // (隣接ノードのID, 隣接ノードまでの距離)
  std::vector<std::vector<unsigned_pair>> adj(3);

  adj[0].push_back(std::make_pair(1U, 3U));
  adj[0].push_back(std::make_pair(2U, 1U));
  adj[2].push_back(std::make_pair(1U, 1U));

  auto const path(shortest_path(adj, 0, 1));
  std::copy(path.begin(), path.end(),
    std::ostream_iterator<unsigned>(std::cout, "\n"));

  return 0;
}


