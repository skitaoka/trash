#pragma once

#ifndef AKA_INCLUDED
#define AKA_INCLUDED

#include <limits>

namespace aka {

	template <typename T>
	inline T sgn(T const x)
	{
		if (x > T()) {
			return T(1);
		}
		if (x < T()) {
			return -T(1);
		}
		return T();
	}

	template <typename T>
	inline T rcp(T const x)
	{
		return T(1) / x;
	}

	template <typename T>
	inline T square_of(T const x)
	{
		return x * x;
	}
}

namespace aka {

	template <typename T>
	class vector3
	{
	public:
		inline vector3()
		{
		}

		inline explicit vector3(T const _s)
		{
			data_[0] = _s;
			data_[1] = _s;
			data_[2] = _s;
		}

		inline vector3(T const _x, T const _y, T const _z)
		{
			data_[0] = _x;
			data_[1] = _y;
			data_[2] = _z;
		}

		inline T   operator [] (int const i) const { return data_[i]; };
		inline T & operator [] (int const i)       { return data_[i]; };

		inline T   x() const { return data_[0]; };
		inline T & x()       { return data_[0]; };
		inline T   y() const { return data_[1]; };
		inline T & y()       { return data_[1]; };
		inline T   z() const { return data_[2]; };
		inline T & z()       { return data_[2]; };

	private:
		T data_[3];
	};

	template <typename T>
	class matrix3
	{
	public:
		inline matrix3()
		{
		}

		// *this = _s I
		inline explicit matrix3(T const _s)
		{
			data_[0] =  _s; data_[1] = T(); data_[2] = T();
			data_[3] = T(); data_[4] =  _s; data_[5] = T();
			data_[6] = T(); data_[7] = T(); data_[8] =  _s;
		}

		// *this = diag(_a, _b, _c)
		inline matrix3(T const _a, T const _b, T const _c)
		{
			data_[0] =  _a; data_[1] = T(); data_[2] = T();
			data_[3] = T(); data_[4] =  _b; data_[5] = T();
			data_[6] = T(); data_[7] = T(); data_[8] =  _c;
		}

		// *this = diag(v.x, v.y, v.z)
		inline explicit matrix3(vector3<T> const _v)
		{
			data_[0] = _v[0]; data_[1] =   T(); data_[2] =   T();
			data_[3] =   T(); data_[4] = _v[1]; data_[5] =   T();
			data_[6] =   T(); data_[7] =   T(); data_[8] = _v[2];
		}

		inline T   operator () (int const i, int const j) const { return data_[i*3+j]; }
		inline T & operator () (int const i, int const j)       { return data_[i*3+j]; }

	private:
		T data_[3*3];
	};

	// *y += x
	template <typename T>
	vector3<T> & add(vector3<T> const & x, vector3<T> * const y)
	{
		(*y)[0] += x[0];
		(*y)[1] += x[1];
		(*y)[2] += x[2];
		return (*y);
	}

	// *y -= x
	template <typename T>
	vector3<T> & sub(vector3<T> const & x, vector3<T> * const y)
	{
		(*y)[0] -= x[0];
		(*y)[1] -= x[1];
		(*y)[2] -= x[2];
		return (*y);
	}

	// *y (*)= x
	template <typename T>
	vector3<T> & mul(vector3<T> const & x, vector3<T> * const y)
	{
		(*y)[0] *= x[0];
		(*y)[1] *= x[1];
		(*y)[2] *= x[2];
		return (*y);
	}

	// *y (/)= x
	template <typename T>
	vector3<T> & div(vector3<T> const & x, vector3<T> * const y)
	{
		(*y)[0] /= x[0];
		(*y)[1] /= x[1];
		(*y)[2] /= x[2];
		return (*y);
	}

	// *y += x * a
	template <typename T>
	vector3<T> & axpy(T const a, vector3<T> const & x, vector3<T> * const y)
	{
		(*y)[0] += x[0] * a;
		(*y)[1] += x[1] * a;
		(*y)[2] += x[2] * a;
		return (*y);
	}

	// *z = y + x * a
	template <typename T>
	vector3<T> & axpyz(T const a, vector3<T> const & x, vector3<T> const & y, vector3<T> * const z)
	{
		(*z)[0] = y[0] + x[0] * a;
		(*z)[1] = y[1] + x[1] * a;
		(*z)[2] = y[2] + x[2] * a;
		return (*z);
	}

	// *y = *y * a + x 
	template <typename T>
	vector3<T> & xpay(vector3<T> const & x, T const a, vector3<T> * const y)
	{
		(*y)[0] = (*y)[0] * a + x[0];
		(*y)[1] = (*y)[1] * a + x[1];
		(*y)[2] = (*y)[2] * a + x[2];
		return (*y);
	}

	// *x *= a
	template <typename T>
	vector3<T> & scale(T const a, vector3<T> * const x)
	{
		(*x)[0] *= a;
		(*x)[1] *= a;
		(*x)[2] *= a;
		return (*x);
	}

	// return x . y
	template <typename T>
	T dot(vector3<T> const & x, vector3<T> const & y)
	{
		return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
	}

	// *z = x x y
	template <typename T>
	vector3<T> & cross(vector3<T> const & x, vector3<T> const & y, vector3<T> * const z)
	{
		(*z)[0] = x[1] * y[2] - x[2] * y[1];
		(*z)[1] = x[2] * y[0] - x[0] * y[2];
		(*z)[2] = x[0] * y[1] - x[1] * y[0];
		return (*z);
	}

	// return x^T y
	template <typename T>
	T inner(vector3<T> const & x, vector3<T> const & y)
	{
		return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
	}

	// *m = x y^T
	template <typename T>
	matrix3<T> & outer(vector3<T> const & x, vector3<T> const & y, matrix3<T> * const m)
	{
		(*m)(0,0) = x[0] * y[0];  (*m)(0,1) = x[0] * y[1];  (*m)(0,2) = x[0] * y[2];
		(*m)(1,0) = x[1] * y[0];  (*m)(1,1) = x[1] * y[1];  (*m)(1,2) = x[1] * y[2];
		(*m)(2,0) = x[2] * y[0];  (*m)(2,1) = x[2] * y[1];  (*m)(2,2) = x[2] * y[2];
		return (*m);
	}

	// *c = a b
	template <typename T>
	matrix3<T> & mul(matrix3<T> const & a, vector3<T> const & b, vector3<T> * const c)
	{
		(*c)[0] = a(0,0) * b[0] + a(0,1) * b[0] + a(0,2) * b[0];
		(*c)[1] = a(0,0) * b[1] + a(0,1) * b[1] + a(0,2) * b[1];
		(*c)[2] = a(0,0) * b[2] + a(0,1) * b[2] + a(0,2) * b[2];

		return (*c);
	}

	// *c = a b
	template <typename T>
	matrix3<T> & mul(matrix3<T> const & a, matrix3<T> const & b, matrix3<T> * const c)
	{
		(*c)(0,0) = a(0,0) * b(0,0) + a(0,1) * b(1,0) + a(0,2) * b(2,0);
		(*c)(0,1) = a(0,0) * b(0,1) + a(0,1) * b(1,1) + a(0,2) * b(2,1);
		(*c)(0,2) = a(0,0) * b(0,2) + a(0,1) * b(1,2) + a(0,2) * b(2,2);

		(*c)(1,0) = a(1,0) * b(0,0) + a(1,1) * b(1,0) + a(1,2) * b(2,0);
		(*c)(1,1) = a(1,0) * b(0,1) + a(1,1) * b(1,1) + a(1,2) * b(2,1);
		(*c)(1,2) = a(1,0) * b(0,2) + a(1,1) * b(1,2) + a(1,2) * b(2,2);

		(*c)(2,0) = a(2,0) * b(0,0) + a(2,1) * b(1,0) + a(2,2) * b(2,0);
		(*c)(2,1) = a(2,0) * b(0,1) + a(2,1) * b(1,1) + a(2,2) * b(2,1);
		(*c)(2,2) = a(2,0) * b(0,2) + a(2,1) * b(1,2) + a(2,2) * b(2,2);

		return (*c);
	}

	// *c = a^T b
	template <typename T>
	matrix3<T> & inner(matrix3<T> const & a, matrix3<T> const & b, matrix3<T> * const c)
	{
		(*c)(0,0) = a(0,0) * b(0,0) + a(1,0) * b(1,0) + a(2,0) * b(2,0);
		(*c)(1,0) = a(0,1) * b(0,0) + a(1,1) * b(1,0) + a(2,1) * b(2,0);
		(*c)(2,0) = a(0,2) * b(0,0) + a(1,2) * b(1,0) + a(2,2) * b(2,0);

		(*c)(0,1) = a(0,0) * b(0,1) + a(1,0) * b(1,1) + a(2,0) * b(2,1);
		(*c)(1,1) = a(0,1) * b(0,1) + a(1,1) * b(1,1) + a(2,1) * b(2,1);
		(*c)(2,1) = a(0,2) * b(0,1) + a(1,2) * b(1,1) + a(2,2) * b(2,1);

		(*c)(0,2) = a(0,0) * b(0,2) + a(1,0) * b(1,2) + a(2,0) * b(2,2);
		(*c)(1,2) = a(0,1) * b(0,2) + a(1,1) * b(1,2) + a(2,1) * b(2,2);
		(*c)(2,2) = a(0,2) * b(0,2) + a(1,2) * b(1,2) + a(2,2) * b(2,2);

		return (*c);
	}

	// *c = a b^T
	template <typename T>
	matrix3<T> & outer(matrix3<T> const & a, matrix3<T> const & b, matrix3<T> * const c)
	{
		(*c)(0,0) = a(0,0) * b(0,0) + a(0,1) * b(0,1) + a(0,2) * b(0,2);
		(*c)(0,1) = a(0,0) * b(1,0) + a(0,1) * b(1,1) + a(0,2) * b(1,2);
		(*c)(0,2) = a(0,0) * b(2,0) + a(0,1) * b(2,1) + a(0,2) * b(2,2);

		(*c)(1,0) = a(1,0) * b(0,0) + a(1,1) * b(0,1) + a(1,2) * b(0,2);
		(*c)(1,1) = a(1,0) * b(1,0) + a(1,1) * b(1,1) + a(1,2) * b(1,2);
		(*c)(1,2) = a(1,0) * b(2,0) + a(1,1) * b(2,1) + a(1,2) * b(2,2);

		(*c)(2,0) = a(2,0) * b(0,0) + a(2,1) * b(0,1) + a(2,2) * b(0,2);
		(*c)(2,1) = a(2,0) * b(1,0) + a(2,1) * b(1,1) + a(2,2) * b(1,2);
		(*c)(2,2) = a(2,0) * b(2,0) + a(2,1) * b(2,1) + a(2,2) * b(2,2);

		return (*c);
	}

	template <typename T>
	matrix3<T> & transpose(matrix3<T> const & a, matrix3<T> * const b)
	{
		(*b)(0,0) = a(0,0); (*b)(0,1) = a(1,0); (*b)(0,2) = a(2,0);
		(*b)(1,0) = a(0,1); (*b)(1,1) = a(1,1); (*b)(1,2) = a(2,1);
		(*b)(2,0) = a(0,2); (*b)(2,1) = a(1,2); (*b)(2,2) = a(2,2);

		return (*b);
	}

	// eigendecomposition using Jacobi iteration.
	// 半正定値実対称行列の固有値分解を求める.
	template <typename T>
	void jacobi(matrix3<T> * const a, matrix3<T> * const r, vector3<T> * const d)
	{
		// 単位行列で初期化
		(*r)(0,0)=T(1); (*r)(0,1)=T(0); (*r)(0,2)=T(0);
		(*r)(1,0)=T(0); (*r)(1,1)=T(1); (*r)(1,2)=T(0);
		(*r)(2,0)=T(0); (*r)(2,1)=T(0); (*r)(2,2)=T(1);

		// 対角要素で初期化
		(*d)[0] = (*a)(0,0);
		(*d)[1] = (*a)(1,1);
		(*d)[2] = (*a)(2,2);

		vector3<int> const ps(0, 0, 1);
		vector3<int> const qs(1, 2, 2);

		for (int n = 0; n < 6; ++n) {
			for (int m = 0; m < 3; ++m) {
				int const p = ps[m];
				int const q = qs[m];

				T const apq = (*a)(p,q);
				if (std::abs(apq) >= std::numeric_limits<T>::epsilon()) {

					T const app = (*d)[p];
					T const aqq = (*d)[q];

					T const cot = (T(1)/2) * (aqq - app) / apq;
					T const t = aka::sgn(cot) / (std::abs(cot) + std::sqrt(T(1) + aka::square_of(cot)));

					T const c = aka::rcp(std::sqrt(T(1) + aka::square_of(t)));
					T const s = t * c;

					T const tau = s / (T(1) + c);

					{
						T const h = t * apq;
						(*d)[p] -= h;
						(*d)[q] += h;
					}

					(*a)(p,q) = T();

#define AKA_ROTATE(mtx,i,j,k,l)\
					{\
						T const x = (mtx)(i,j);\
						T const y = (mtx)(k,l);\
						(mtx)(i,j) = x - s * (y + x * tau);\
						(mtx)(k,l) = y + s * (x - y * tau);\
					}
					for (int j =   0; j < p; ++j) AKA_ROTATE(*a, j, p, j, q);
					for (int j = p+1; j < q; ++j) AKA_ROTATE(*a, p, j, j, q);
					for (int j = q+1; j < 3; ++j) AKA_ROTATE(*a, p, j, q, j);
					for (int j =   0; j < 3; ++j) AKA_ROTATE(*r, j, p, j, q);
#undef AKA_ROTATE
				} else {
					(*a)(p,q) = T();
				}
			}
		}
	}

	// SVD (singular value decomposition) based on eigendecomposition using Jacobi iteration.
	// svd(a) = u s v^T
	template <typename T>
	void svd(aka::matrix3<T> const & a, aka::matrix3<T> * const u, aka::vector3<T> * const s, aka::matrix3<T> * const vt)
	{
		// b := a a^T
		aka::matrix3<T> b;
		aka::outer(a, a, &b);

		// 任意の実数行列 a の a a^T = a^T a は 半正定値実対称行列
		// b -> u s u^T
		aka::jacobi(&b, u, s);

		// singular value
		(*s)[0] = std::sqrt((*s)[0]);
		(*s)[1] = std::sqrt((*s)[1]);
		(*s)[2] = std::sqrt((*s)[2]);

		// v^{T} = s^{-1} u^T a
#if 0
		aka::matrix3<T> c;
		aka::outer(aka::matrix3<T>(
			((*s)[0] > T()) ? aka::rcp((*s)[0]) : T(),
			((*s)[1] > T()) ? aka::rcp((*s)[1]) : T(),
			((*s)[2] > T()) ? aka::rcp((*s)[2]) : T()), *u, &c);
		aka::mul(c, a, vt);
#else
		aka::inner(*u, a, vt);
		{ T const r = ((*s)[0] > T()) ? aka::rcp((*s)[0]) : T(); (*vt)(0,0) *= r; (*vt)(0,1) *= r; (*vt)(0,2) *= r; }
		{ T const r = ((*s)[1] > T()) ? aka::rcp((*s)[1]) : T(); (*vt)(1,0) *= r; (*vt)(1,1) *= r; (*vt)(1,2) *= r; }
		{ T const r = ((*s)[2] > T()) ? aka::rcp((*s)[2]) : T(); (*vt)(2,0) *= r; (*vt)(2,1) *= r; (*vt)(2,2) *= r; }
#endif
	}
}

#endif//AKA_INCLUDED
