///
/// Binary Indexed Tree (Fenwick Tree) による累積頻度表の実装.
/// cf.  http://d.hatena.ne.jp/naoya/20090606/1244284915
///
namespace cute
{
	namespace algorithm
	{
		// データを二分木で管理して、その範囲の計算をした値を保持する。
		template <typename T, std::size_t Sz, typename Op>
		class binary_indexed_tree
		{
		public:
			binary_indexed_tree()
			{
			}

			T get(std::size_t idx)
			{
				assert(idx <= Sz);

				T sum = 0;
				while (idx) {
					Op::apply(sum, data_[idx]);
					idx -= idx & -idx; // 上位ビットから順に
				}
				return sum;
			}

			void add(std::size_t idx, T val)
			{
				assert(idx <= Sz);

				while (idx <= Sz) {
					Op::apply(data_[idx], val);
					idx += idx & -idx; // 下位ビットから順に
				}
			}

		private:
			// [0] はダミー要素
			std::array<T, Sz+1> data_;
		};
	}//end of namespace algorithm
}//end of namespace cute

