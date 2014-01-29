#pragma once

#ifndef AKA_REAL_INCLUDED
#define AKA_REAL_INCLUDED

#include <cassert>

#ifndef BOOST_NONCOPYABLE_HPP_INCLUDED
#include <boost/noncopyable.hpp>
#endif

#define AKA_INLINE __forceinline __device__ __host__

namespace aka {
	namespace mpm {

// 代入系のファンクタ（二項演算子類として実装）
#define IMPLEMENT_MACRO(Name, Op) \
		template <typename T>\
		struct Name: boost::noncopyable\
		{\
			typedef void value_type;\
			static AKA_INLINE value_type\
				apply_on(T & lhs, T const rhs)\
			{\
				lhs Op rhs;\
			}\
		}
		IMPLEMENT_MACRO(fcnl_Assign,  =);
		IMPLEMENT_MACRO(fcnl_AddUpd, +=);
		IMPLEMENT_MACRO(fcnl_SubUpd, -=);
		IMPLEMENT_MACRO(fcnl_MulUpd, *=);
		IMPLEMENT_MACRO(fcnl_DivUpd, /=);
#undef IMPLEMENT_MACRO

// 二項演算系のファンクタ
#define IMPLEMENT_MACRO(Name, Op)\
		template <typename T>\
		struct Name: boost::noncopyable\
		{\
			typedef T value_type;\
			static AKA_INLINE value_type\
				apply_on(value_type const lhs, value_type const rhs)\
			{\
				return lhs Op rhs;\
			}\
		}
		IMPLEMENT_MACRO(fcnl_Add, +);
		IMPLEMENT_MACRO(fcnl_Sub, -);
		IMPLEMENT_MACRO(fcnl_Mul, *);
		IMPLEMENT_MACRO(fcnl_Div, /);
#undef IMPLEMENT_MACRO

// 単項演算系のファンクタ
#define IMPLEMENT_MACRO(Name, Op)\
		template <typename T>\
		struct Name: boost::noncopyable\
		{ \
			typedef T value_type;\
			static AKA_INLINE value_type\
				apply_on(value_type const rhs)\
			{\
				return Op rhs;\
			}\
		}
		IMPLEMENT_MACRO(fcnl_Negate, -);
#undef IMPLEMENT_MACRO
	}
}

namespace aka {
	namespace mpm {

		template <typename T, int Sz>
		class real_t;

		// ET; expression template
		namespace details {

			//! node of expression template.
			template <typename E, int Sz>
			class node_t
			{
			public:
				typedef          E             expr1_type;
				typedef typename E::value_type value_type;

			public:
				AKA_INLINE explicit node_t(expr1_type const & expr1)
					: expr1_(expr1)
				{
				}

				AKA_INLINE value_type operator [] (int const i) const
				{
					assert((i >= 0) || !"");
					assert((i < Sz) || !"");
					return expr1_[i];
				}

			private:
				expr1_type expr1_;
			};

			//! 数式テンプレートのノードに入ってる参照を表すもの
			template <typename T, int Sz>
			class cref_t
			{
			public:
				typedef T value_type;

			public:
				AKA_INLINE cref_t(value_type const * const a)
					: data_(a)
				{
				}

				AKA_INLINE value_type operator [] (int const i) const
				{
					assert((i >= 0) || !"");
					assert((i < Sz) || !"");
					return data_[i];
				}

			private:
				value_type const * const data_;
			};

			//! 数式テンプレートのノードに入ってる単項演算を表すもの
			template <typename Op, typename E1>
			class unary_operator
			{
			public:
				typedef typename Op::return_type value_type;
				typedef          E1              expr1_type;

			public:
				AKA_INLINE explicit unary_operator(expr1_type const & expr1)
					: expr1_(expr1)
				{
				}

			public:
				AKA_INLINE value_type operator [] (int const i) const
				{
					return Op::apply_on(expr1_[i]);
				}

			private:
				expr1_type expr1_;
			};

			//! 数式テンプレートのノードに入ってる２項演算を表すもの
			template <typename Op, typename E1, typename E2>
			class binary_operator
			{
			public:
				typedef typename Op::value_type value_type;
				typedef          E1             expr1_type;
				typedef          E2             expr2_type;

			public:
				AKA_INLINE binary_operator(expr1_type const & expr1, expr2_type const & expr2)
					: expr1_(expr1)
					, expr2_(expr2)
				{
				}

			public:
				AKA_INLINE value_type operator [] (int const i) const
				{
					return Op::apply_on(expr1_[i], expr2_[i]);
				}

			private:
				expr1_type expr1_;
				expr2_type expr2_;
			};
		}
	}
}

namespace aka {
	namespace mpm {

		//! soa (structure of arrays) と aos (array of structures) のミックス
		template <typename T, int Sz>
		class real_t
		{
		public:
			typedef T value_type;

		public:
			T operator [] (int const i) const
			{
				return data_[i];
			}

			T & operator [] (int const i)
			{
				return data_[i];
			}

			T const * data() const
			{
				return data_;
			}

			T * data()
			{
				return data_;
			}

		public:
			AKA_INLINE explicit real_t(T const x = T())
			{
				std::fill_n(data_, Sz, x);
			}

			AKA_INLINE real_t(real_t<T, Sz> const & v)
			{
				assign<real_t<T, Sz>, fcnl_Assign<value_type>>(v);
			}

			template <typename E>
			AKA_INLINE explicit real_t(aka::mpm::details::node_t<E, Sz> const & v)
			{
				assign<aka::mpm::details::node_t<E, Sz>, fcnl_Assign<value_type>>(v);
			}

#define IMPLEMENT_MACRO(Name, Op)\
			AKA_INLINE real_t<value_type, Sz> & operator Op (real_t<value_type, Sz> const & v)\
			{\
				assign<real_t<value_type, Sz>, Name<value_type>>(v);\
				return *this;\
			}\
			template <typename E>\
			AKA_INLINE real_t<value_type, Sz> & operator Op (aka::mpm::details::node_t<E, Sz> const & v)\
			{\
				assign<aka::mpm::details::node_t<E, Sz>, Name<value_type>>(v);\
				return *this;\
			}
			IMPLEMENT_MACRO(fcnl_Assign,  =)
			IMPLEMENT_MACRO(fcnl_AddUpd, +=)
			IMPLEMENT_MACRO(fcnl_SubUpd, -=)
			IMPLEMENT_MACRO(fcnl_MulUpd, *=)
			IMPLEMENT_MACRO(fcnl_DivUpd, /=)
#undef IMPLEMENT_MACRO

		private:
			template <typename E, typename Fn>
			AKA_INLINE void assign(E const & e)
			{
#ifdef _MSC_VER
#pragma loop(ivdep)
#endif
				for (int i = 0; i < Sz; ++i) {
					Fn::apply_on(data_[i], e[i]);
				}
			}

		private:
			value_type data_[Sz];
		};
	}
}

//
// operator(real_t<S, Sz>, real_t<T, Sz>)
// operator(real_t<T, Sz>, node_t<E, Sz>)
// operator(node_t<E, Sz>, real_t<T, Sz>)
// operator(node_t<E, Sz>, node_t<F, Sz>)
//
#define IMPLEMENT_MACRO(Name, Op)\
	template <typename T, int Sz>\
	AKA_INLINE\
	aka::mpm::details::node_t<\
		aka::mpm::details::binary_operator<\
			Name<T>,\
			aka::mpm::details::cref_t<T, Sz>,\
			aka::mpm::details::cref_t<T, Sz>>, Sz>\
	operator Op (\
		aka::mpm::real_t<T, Sz> const & lhs,\
		aka::mpm::real_t<T, Sz> const & rhs)\
	{\
		typedef aka::mpm::details::binary_operator<\
			Name<T>,\
			aka::mpm::details::cref_t<T, Sz>,\
			aka::mpm::details::cref_t<T, Sz>>\
		expression_type;\
		return aka::mpm::details::node_t<expression_type, Sz>(\
			expression_type(\
				aka::mpm::details::cref_t<T, Sz>(lhs.data()),\
				aka::mpm::details::cref_t<T, Sz>(rhs.data())));\
	}\
	\
	template <typename E, int Sz>\
	AKA_INLINE\
	aka::mpm::details::node_t<\
		aka::mpm::details::binary_operator<\
			Name<typename E::value_type>,\
			aka::mpm::details::node_t<         E            , Sz>,\
			aka::mpm::details::cref_t<typename E::value_type, Sz>>, Sz>\
	operator Op (\
		aka::mpm::details::node_t<         E            , Sz> const & lhs,\
		aka::mpm::         real_t<typename E::value_type, Sz> const & rhs)\
	{\
		typedef aka::mpm::details::binary_operator<\
			Name<typename E::value_type>,\
			aka::mpm::details::node_t<         E            , Sz>,\
			aka::mpm::details::cref_t<typename E::value_type, Sz>>\
		expression_type;\
		return aka::mpm::details::node_t<expression_type, Sz>(\
			expression_type(lhs, aka::mpm::details::cref_t<typename E::value_type, Sz>(rhs.data())));\
	}\
	\
	template <typename E, int Sz>\
	AKA_INLINE\
	aka::mpm::details::node_t<\
		aka::mpm::details::binary_operator<\
			Name<typename E::value_type>,\
			aka::mpm::details::cref_t<typename E::value_type, Sz>,\
			aka::mpm::details::node_t<         E            , Sz>>, Sz>\
	operator Op (\
		aka::mpm::         real_t<typename E::value_type, Sz> const & lhs,\
		aka::mpm::details::node_t<         E            , Sz> const & rhs)\
	{\
		typedef aka::mpm::details::binary_operator<\
			Name<typename E::value_type>,\
			aka::mpm::details::cref_t<typename E::value_type, Sz>,\
			aka::mpm::details::node_t<         E            , Sz>>\
		expression_type;\
		return aka::mpm::details::node_t<expression_type, Sz>(\
			expression_type(aka::mpm::details::cref_t<typename E::value_type, Sz>(lhs.data()), rhs));\
	}\
	\
	template <typename E1, typename E2, int Sz>\
	AKA_INLINE\
	aka::mpm::details::node_t<\
		aka::mpm::details::binary_operator<\
			Name<typename E1::value_type>,\
			aka::mpm::details::node_t<E1, Sz>,\
			aka::mpm::details::node_t<E2, Sz>>, Sz>\
	operator Op (\
			aka::mpm::details::node_t<E1, Sz> const & n1,\
			aka::mpm::details::node_t<E2, Sz> const & n2)\
	{\
		typedef aka::mpm::details::binary_operator<\
			Name<typename E2::value_type>,\
			aka::mpm::details::node_t<E1, Sz>,\
			aka::mpm::details::node_t<E2, Sz>>\
		expression_type;\
		return aka::mpm::details::node_t<expression_type, Sz>(\
			expression_type(n1, n2));\
	}
	IMPLEMENT_MACRO(aka::mpm::fcnl_Add, +)
	IMPLEMENT_MACRO(aka::mpm::fcnl_Sub, -)
	IMPLEMENT_MACRO(aka::mpm::fcnl_Mul, *)
	IMPLEMENT_MACRO(aka::mpm::fcnl_Div, /)
#undef IMPLEMENT_MACRO

//
// unary_operator(real_t<T, Sz>)
// unary_operator(node_t<F, Sz>)
//
#define IMPLEMENT_MACRO(Name, Op)\
	template <typename T, int Sz>\
	AKA_INLINE\
	aka::mpm::details::node_t<\
		aka::mpm::details::unary_operator<\
			Name<T>,\
			aka::mpm::details::cref_t<T, Sz>>, Sz>\
	operator Op (aka::mpm::real_t<T, Sz> const & rhs)\
	{\
		typedef aka::mpm::details::unary_operator<\
			Name<T>,\
			aka::mpm::details::cref_t<T, Sz>>\
		expression_type;\
		return aka::mpm::details::node_t<expression_type, Sz>(\
			expression_type(aka::mpm::details::cref_t<T, Sz>(rhs.data())));\
	}\
	\
	template <typename E, int Sz>\
	AKA_INLINE\
	aka::mpm::details::node_t<\
		aka::mpm::details::unary_operator<\
			Name<typename E::value_type>,\
			aka::mpm::details::node_t<E, Sz>>, Sz>\
	operator Op (aka::mpm::details::node_t<E, Sz> const & rhs)\
	{\
		typedef aka::mpm::details::unary_operator<\
			Name<typename E::value_type>,\
			aka::mpm::details::node_t<E, Sz>>\
		expression_type;\
		return aka::mpm::details::node_t<expression_type, Sz>(\
			expression_type(rhs));\
	}
	IMPLEMENT_MACRO(aka::mpm::fcnl_Negate, -)
#undef IMPLEMENT_MACRO

#endif//AKA_REAL_INCLUDED
