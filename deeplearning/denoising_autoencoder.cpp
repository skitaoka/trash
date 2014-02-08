#include "aka/math.hpp"

namespace aka
{
	template <typename T>
	class denoising_autoencoder
	{
	public:
		denoising_autoencoder(std::size_t const _n, std::size_t const _m, std::mt19937 & engine)
			: n_(_n)
			, m_(_m)
			, epoch_(0)
			, weights_(_n * _m)
			, offsetA_(_m)
			, offsetB_(_n)
		{
			std::normal_distribution<T> dist(T(), T(0.01));
			for (std::size_t k = 0, size = weights_.size(); k < size; ++k) {
				weights_[k] = dist(engine);
			}
			for (std::size_t j = 0; j < _m; ++j) {
				offsetA_[j] = dist(engine);
			}
			for (std::size_t i = 0; i < _n; ++i) {
				offsetB_[i] = dist(engine);
			}
		}

		std::vector<T> operator () (std::vector<T> const & x) const
		{
			std::vector<T> y(m_);
			for (std::size_t j = 0; j < m_; ++j) {
				T f = T();
				for (std::size_t i = 0; i < n_; ++i) {
					f += weights_[j*n_+i] * x[i];
				}
				y[j] = aka::sigmoid(f + offsetA_[j]);
			}
		}

		void learn(std::vector<std::vector<T>> const & x, T const eta, T const p, std::mt19937 & engine)
		{
			// ���z
			std::vector<T> dw(n_*m_);
			std::vector<T> da(m_);
			std::vector<T> db(n_);

			// ���ԕϐ�
			std::vector<T> x_noized(n_);
			std::vector<T> y(m_);
			std::vector<T> z(n_);
			std::vector<T> xmz(n_);//= x-z
			std::vector<T> y1my_wxmz(m_);//= y * (1-y) * (W (x-z))

			std::bernoulli_distribution dist(p);

			for (std::size_t k = 0, size = x.size(); k < size; ++k) {
				for (std::size_t i = 0; i < n_; ++i) {
					x_noized[i] = dist(engine) ? T(1) - x[k][i] : x[k][i];
				}

				for (std::size_t j = 0; j < m_; ++j) {
					T f = T();
					for (std::size_t i = 0; i < n_; ++i) {
						f += weights_[j*n_+i] * x_noized[i];
					}
					y[j] = aka::sigmoid(f + offsetA_[j]);
				}

				for (std::size_t i = 0; i < n_; ++i) {
					T h = T();
					for (std::size_t j = 0; j < m_; ++j) {
						h += weights_[j*n_+i] * y[j];
					}
					z[i] = aka::sigmoid(h + offsetB_[i]);
				}

				// x-z
				for (std::size_t i = 0; i < n_; ++i) {
					xmz[i] = x[k][i] - z[i];
				}

				// y * (1-y) * (W (x-z))
				for (std::size_t j = 0; j < m_; ++j) {
					T a = T();
					for (std::size_t i = 0; i < n_; ++i) {
						a += weights_[j*n_+i] * xmz[i];
					}
					y1my_wxmz[j] = y[j] * (1 - y[j]) * a;
				}

				for (std::size_t j = 0; j < m_; ++j) {
					for (std::size_t i = 0; i < n_; ++i) {
						dw[j*n_+i] += y1my_wxmz[j] * x_noized[i] + y[j] * xmz[i];
					}
				}
				for (std::size_t j = 0; j < m_; ++j) {
					da[j] += y1my_wxmz[j];
				}
				for (std::size_t i = 0; i < n_; ++i) {
					db[i] += xmz[i];
				}
			}

			// ���z�ɏ]���ďd�݂��X�V.
			T const alpha = eta / ++epoch_;
			for (std::size_t k = 0, size = dw.size(); k < size; ++k) {
				weights_[k] = weights_[k] + alpha * dw[k];
			}
			for (std::size_t j = 0; j < m_; ++j) {
				offsetA_[j] = offsetA_[j] + alpha * da[j];
			}
			for (std::size_t i = 0; i < n_; ++i) {
				offsetB_[i] = offsetB_[i] + alpha * db[i];
			}
		}

		void show(std::vector<std::vector<T>> const & x)
		{
			std::vector<T> y(m_);
			for (std::size_t k = 0, size = x.size(); k < size; ++k) {
				std::cout << "input\t:";
				for (std::size_t i = 0; i < n_; ++i) {
					std::cout << '\t' << x[k][i];
				}
				std::cout << '\n';
				
				for (std::size_t j = 0; j < m_; ++j) {
					T f = T();
					for (std::size_t i = 0; i < n_; ++i) {
						f += weights_[j*n_+i] * x[k][i];
					}
					y[j] = aka::sigmoid(f + offsetA_[j]);
				}
				
				std::cout << "output\t:";
				for (std::size_t i = 0; i < n_; ++i) {
					T h = T();
					for (std::size_t j = 0; j < m_; ++j) {
						h += weights_[j*n_+i] * y[j];
					}
					T const zi = aka::sigmoid(h + offsetB_[i]);
					std::cout << '\t' << aka::is_true(zi);
				}
				std::cout << '\n';

				std::cout << "hidden\t:";
				for (std::size_t j = 0; j < m_; ++j) {
					std::cout << '\t' << y[j];
				}

				std::cout << "\n\n";
			}
		}

	private:
		std::size_t const n_; // ���͑w�̎���
		std::size_t const m_; // �o�͑w�̎���
		std::size_t    epoch_;
		std::vector<T> weights_;
		std::vector<T> offsetA_;
		std::vector<T> offsetB_;
	};
}

int main(int argc, char* argv[])
{
	std::ios::sync_with_stdio(false);

	std::random_device device;
	std::mt19937 engine(device());

	// input example:
	//  10 1 10000 0.5 0.3
	//  2
	//  0 0 0 0 0 1 1 1 1 1
	//  1 1 1 1 1 0 0 0 0 0

	std::size_t n;    std::cin >> n;    // ���͎���
	std::size_t m;    std::cin >> m;    // �o�͎���
	std::size_t itrs; std::cin >> itrs; // ������
	double      eta;  std::cin >> eta;  // �w�K��
	double      p;    std::cin >> p;    // �h���b�v��
	std::size_t size; std::cin >> size; // �f�[�^�T�C�Y

	std::vector<std::vector<double>> x(size);
	for (std::size_t k = 0; k < size; ++k) {
		x[k].resize(n);
		for (std::size_t i = 0; i < n; ++i) {
			std::cin >> x[k][i];
		}
	}

	aka::denoising_autoencoder<double> da(n, m, engine);
	for (std::size_t epoch = 0; epoch < itrs; ++epoch) {
		da.learn(x, eta, p, engine);
	}
	da.show(x);

	return 0;
}