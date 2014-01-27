#include "aka/math.hpp"

namespace aka
{
	template <typename T>
	class maxout
	{
	public:
		inline maxout(
			std::size_t const _m,
			std::size_t const _d = 5,
			std::size_t const _n = 1)
			: m_(_m)
			, d_(_d)
			, n_(_n)
			, weights_((_m+1) * _d * _n)
			, outputs_(_n)
			,	indices_(_n)
			, counts_ (_d)
			,	deltaIs_((_m+1))
		{
		}

		void init(std::mt19937 & engine)
		{
			std::bernoulli_distribution dist;
			for (std::size_t i = 0, size = weights_.size(); i < size; ++i) {
				weights_[i] = 2.0 * dist(engine) - 1.0;
			}
		}

		// @param x m-dimensional input vector.
		template <bool EnableDropout, typename S>
		std::vector<T> const &
			forward(std::vector<S> const & x)
		{
			std::size_t const m = m_;
			std::size_t const n = n_;
			std::size_t const d = d_;

			std::vector<bool> dropout(d, true);
			if (EnableDropout) {
				std::fill_n(dropout.begin(), 3, false);
			}

			for (std::size_t i = 0; i < n; ++i) {
				T           max_value = -std::numeric_limits<T>::infinity();
				std::size_t max_index = 0;

				if (EnableDropout) {
					std::random_shuffle(dropout.begin(), dropout.end());
				}

				for (std::size_t k = 0; k < d; ++k) {
					if (EnableDropout && dropout[k]) {
						continue;
					}

					T value = T();
					for (std::size_t j = 0; j < m; ++j) {
						std::size_t const idx = (i*d+k)*(m+1)+j;
						value += weights_[idx] * x[j];
					}
					{
						std::size_t const idx = (i*d+k)*(m+1)+m;
						value += weights_[idx];
					}

					if (max_value < value) {
						max_value = value;
						max_index = k;
					}
				}

				outputs_[i] = max_value;
				indices_[i] = max_index;
			}

			return outputs_;
		}

		// @param x a input vector.
		// @param deltaIs dE/df
		// @param eta a learn rate.
		template <typename S>
		void backword(
			std::vector<S> const & x,
			std::vector<T> const & deltaIs,
			T const alpha = 0.5,
			T const beta  = 0.999995)
		{
			std::size_t const m = m_;
			std::size_t const n = n_;
			std::size_t const d = d_;

			for (std::size_t i = 0; i < n; ++i) {
				std::size_t const k = indices_[i];
				std::size_t const c = counts_ [k]; ++counts_[k];
				T const alpha_delta = alpha * deltaIs[i];
				for (std::size_t j = 0; j < m; ++j) {
					std::size_t const idx = (i*d+k)*(m+1)+j;
					double const w_old = weights_[idx];
					double const w_new = beta * (w_old - alpha_delta * x[j]);
					weights_[idx] = (c * w_old + w_new) / (c+1);
				}
				{
					std::size_t const idx = (i*d+k)*(m+1)+m;
					double const w_old = weights_[idx];
					double const w_new =        (w_old - alpha_delta); // 切片は正則化しない
					weights_[idx] = (c * w_old + w_new) / (c+1);
				}
			}
		}

		std::vector<T> const & outputs() const { return outputs_; }
		std::vector<T> const & deltaIs() const { return deltaIs_; }

	private:
		std::size_t const m_; // 入力の次元
		std::size_t const d_; // 内部の次数
		std::size_t const n_; // 出力の次元

		std::vector<T>           weights_; // 重み
		std::vector<T>           outputs_; // 出力値
		std::vector<std::size_t> indices_; // 有効なインデックス
		std::vector<std::size_t> counts_ ; // 更新回数
		std::vector<T>           deltaIs_; // 誤差関数をこのレイヤーの関数で微分した値
	};
}


namespace aka
{
	template <typename T>
	class svm
	{
	public:
		svm(std::vector<std::vector<T>> v)
			: v_(v)
			, w_(v_.size()+1)
			, c_(0)
		{
		}

		template <typename K>
		T operator () (K const kernel, std::vector<T> const & x) const
		{
			T retval = T();

			std::size_t const n = v_.size();
			for (std::size_t i = 0; i < n; ++i) {
				retval += w_[i] * kernel(x, v_[i]);
			}
			{
				retval += w_[n];
			}

			return retval;
		}

		template <typename K>
		void update(K const kernel,
			std::vector<T> const & x,
			T const t,
			T const alpha = 0.5,
			T const beta  = 0.999995)
		{
			std::size_t const n = v_.size();

#if 0
			T y = T();
			{
				std::vector<bool> use(n);
				for (std::size_t i = n/2; i < n; ++i) {
					use[i] = true;
				}

				std::size_t const n = v_.size();
				for (std::size_t i = 0; i < n; ++i) {
					if (use[i]) {
						continue;
					}
					y += w_[i] * kernel(x, v_[i]);
				}

				double const p = std::accumulate(use.begin(), use.end(), 0.0, std::plus<double>())/n;
				y /= p;

				{
					y += w_[n];
				}
			}
#else
			T const y = (*this)(kernel, x);
#endif

			if (t * y < 1.0) {
				std::size_t const c = c_; ++c_;
				for (std::size_t i = 0; i < n; ++i) {
					T const w_old = w_[i];
					T const w_new = beta * (w_old + alpha * t * kernel(x, v_[i]));
					w_[i] = (c * w_old + w_new) / (c+1);
				}
				{
					T const w_old = w_[n];
					T const w_new =        (w_old + alpha * t); // 切片は正則化しない
					w_[n] = (c * w_old + w_new) / (c+1);
				}
			}
		}

	private:
		std::vector<std::vector<T>> v_; // support vectors
		std::vector<T>              w_; // weights
		std::size_t                 c_; // 更新回数
	};
}

int main(int argc, char* argv[])
{
	std::random_device device;
	std::mt19937 engine(device());

	std::vector<double> x(2); // 入力ベクトル
	bool const data[][2] = {
		{0,0},
		{1,0},
		{0,1},
		{1,1},
	};
	int const data_size = sizeof(data)/sizeof(data[0]);

#if 0
	aka::maxout<double> h(x.size(), 9, 1);
	std::vector<double> deltaIs(1);
	for (int epic = 0; epic < 1000; ++epic) {
		for (int n = 0; n < data_size; ++n) {
			// 入力データを作る
			x[0] = data[n][0];
			x[1] = data[n][1];

#if 1
			//
			// svm (hinge loss)
			//
			// 教師データを作る
			double const t = aka::expand<double>(aka::xor(aka::is_true(x[0]), aka::is_true(x[1])));

			// 計算してみる
			double const z = h.forward<true>(x)[0];

			// 最小化
			if (t*z < 1.0) {
				deltaIs[0] = -t;
				h.backword(x, deltaIs);
			}
#else
			//
			// logistic regression
			//
			// 教師データを作る
			double const t = aka::xor(aka::is_true(x[0]), aka::is_true(x[1]));

			// 計算してみる
			double const z = aka::sigmoid(h.forward<true>(x)[0]);

			// 最小化
			deltaIs[0] = z - t;
			h.backword(x, deltaIs);
#endif
		}
	}

	for (int n = 0; n < data_size; ++n) {
		// 入力データを作る
		x[0] = data[n][0];
		x[1] = data[n][1];

#if 1
		//
		// svm (hinge loss)
		//
		// 教師データを作る
		double const t = aka::expand<double>(aka::xor(aka::is_true(x[0]), aka::is_true(x[1])));

		// 計算してみる
		double const z = h.forward<true>(x)[0];

		std::cout << x[0] << x[1] << '\t' << t << ':' << z << '\t' << (aka::is_positive(t) == aka::is_positive(z)) << '\n';
#else
		//
		// logistic regression
		//
		// 教師データを作る
		double const t = aka::xor(aka::is_true(x[0]), aka::is_true(x[1]));

		// 計算してみる
		double const z = aka::sigmoid(h.forward<true>(x)[0]);

		std::cout << x[0] << x[1] << '\t' << t << ':' << z << '\t' << (aka::is_true(t) == aka::is_true(z)) << '\n';
#endif
	}
#else
	std::vector<std::vector<double>> support_vectors;
	support_vectors.resize(10);
	std::uniform_real_distribution<double> dist(-1.0, 1.0);
	for (std::size_t i = 0, size = support_vectors.size(); i < size; ++i) {
		support_vectors[i].push_back(dist(engine));
		support_vectors[i].push_back(dist(engine));
	}

	aka::svm<double> machine(support_vectors);

	auto const polynomial_kernel = [](
		std::vector<double> const & x,
		std::vector<double> const & v) -> double
	{
		// (x^T v + offest) ^ 10
		double const offset = 1.0;

		double dot = 0.0;
		for (std::size_t i = 0, size = x.size(); i < size; ++i) {
			dot += x[i] * v[i];
		}
		return aka::pow<10>(dot + offset);
	};

	auto const gussian_kernel = [](
		std::vector<double> const & x,
		std::vector<double> const & v) -> double
	{
		// exp(sigma |x-v|^2)
		double const beta  = 0.5;
		double const sigma = -aka::rsquare(beta);

		double distanceSquared = 0.0;
		for (std::size_t i = 0, size = x.size(); i < size; ++i) {
			distanceSquared += aka::square(x[i] - v[i]);
		}
		return std::exp(sigma * distanceSquared);
	};

	auto const kernel = gussian_kernel;

	for (int epic = 0; epic < 1000; ++epic) {
		for (int n = 0; n < data_size; ++n) {
			// 入力データを作る
			x[0] = data[n][0];
			x[1] = data[n][1];

			// 教師データを作る
			double const t = aka::expand<double>(aka::xor(aka::is_true(x[0]), aka::is_true(x[1])));

			machine.update(kernel, x, t);
		}
	}

	for (int n = 0; n < data_size; ++n) {
		// 入力データを作る
		x[0] = data[n][0];
		x[1] = data[n][1];

		// 教師データを作る
		double const t = aka::expand<double>(aka::xor(aka::is_true(x[0]), aka::is_true(x[1])));

		// 計算してみる
		double const z = machine(kernel, x);

		std::cout << x[0] << x[1] << '\t' << t << ':' << z << '\t' << (aka::is_positive(t) == aka::is_positive(z)) << '\n';
	}
#endif

return 0;
}
