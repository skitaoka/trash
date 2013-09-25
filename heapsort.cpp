//
// Wikipedia のヒープソートがきれいだったので実装してみる．
//
template <typename T>
void make_heap(T data[], std::size_t max, std::size_t i)
{
  std::size_t const j = (i<<1) + 1; // 左の子
  std::size_t const k = (i<<1) + 2; // 右の子

  if (j >= max) {
    return; // 一つも子がない
  }

  if (k == max) {
    // 子が一つしかない
    if (data[i] < data[j]) {
      // 子の方が大きかったら交換
      std::swap(data[i], data[j]);
    }
    return;
  }

  // 再帰的に構築
  make_heap(data, max, j);
  make_heap(data, max, k);

  // 大きい方の子と交換
  std::size_t const d = (data[j] < data[k]) ? k : j;

  if (data[i] < data[d]) {
    // 子の方が大きかったので交換
    std::swap(data[i], data[d]);

    // ヒープ構造が崩れたかもしれないので再構築
    make_heap(data, max, d);
  }
}

template <typename T>
void heap_sort(T data[], std::size_t size)
{
  while (size > 1) {
    make_heap(data, size, 0);
    std::swap(data[0], data[--size]);
  }
}
