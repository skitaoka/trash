///
/// Binary Indexed Tree (Fenwick Tree) �ɂ��ݐϕp�x�\�̎���.
/// cf.  http://d.hatena.ne.jp/naoya/20090606/1244284915
///
namespace cute
{
	namespace algorithm
	{
		// �f�[�^��񕪖؂ŊǗ����āA���͈̔͂̌v�Z�������l��ێ�����B
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
					idx -= idx & -idx; // ��ʃr�b�g���珇��
				}
				return sum;
			}

			void add(std::size_t idx, T val)
			{
				assert(idx <= Sz);

				while (idx <= Sz) {
					Op::apply(data_[idx], val);
					idx += idx & -idx; // ���ʃr�b�g���珇��
				}
			}

		private:
			// [0] �̓_�~�[�v�f
			std::array<T, Sz+1> data_;
		};
	}//end of namespace algorithm
}//end of namespace cute

