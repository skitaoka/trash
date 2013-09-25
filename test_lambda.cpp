/*
 * ラムダの実装
 */
#include <iostream>

//
// 入出力は全てintの１変数関数を普通の算術式の
// 形で記述できるようにするライブラリ。
//  lambda や phoenix を見て自分でも簡単なもの
// を作ってみたくなったので、試しに作りました。
//

namespace petit_fex {

	// int型定数 -----------------------------------
	struct val
	{
		const int v_;
		val( int v ) : v_(v) {}
		int operator () ( int t ) const { return v_; }
	};

	// int型 => val型へのフィルタ ------------------
	template<typename V> struct tr      { typedef  V  typ; };
	template<>           struct tr<int> { typedef val typ; };

	// 変数 ----------------------------------------
	struct var
	{
		int operator () ( int t ) const { return t; }
	};

	// 単項演算子 ----------------------------------
	#define unary_op( NAME, OP )                            \
			template <typename A>                               \
			struct NAME {                                       \
				A a_;                                             \
				NAME(A a) : a_(a) {}                              \
				int operator()(int t) { return OP a_(t); }        \
			};                                                  \
			template<typename A>                                \
			NAME<tr<A>::typ> operator OP ( A a ) {              \
				return NAME<tr<A>::typ>(tr<A>::typ(a));           \
			}

		unary_op( pf_not, ! )
		unary_op( pf_neg, - )
		unary_op( pf_inv, ~ )

	#undef unary_op

	// 二項演算子 ----------------------------------
	#define binary_op( NAME, OP )                           \
			template<typename A, typename B> struct NAME {      \
				NAME(A a, B b) : a_(a), b_(b) {} A a_; B b_;      \
				int operator()(int t) { return a_(t) OP b_(t); }  \
			};                                                  \
			template<typename A, typename B>                    \
			NAME<tr<A>::typ,tr<B>::typ> operator OP (A a, B b)  \
			{                                                   \
				return NAME<tr<A>::typ,tr<B>::typ>                \
						(tr<A>::typ(a),tr<B>::typ(b));                \
			}

		binary_op( pf_plus , +  )
		binary_op( pf_minus, -  )
		binary_op( pf_mult , *  )
		binary_op( pf_div  , /  )
		binary_op( pf_mod  , %  )
		binary_op( pf_lsft , << )
		binary_op( pf_rsft , >> )
		binary_op( pf_eq   , == )
		binary_op( pf_neq  , != )
		binary_op( pf_less , <  )
		binary_op( pf_more , >  )
		binary_op( pf_lesse, <= )
		binary_op( pf_moree, >= )
		binary_op( pf_and  , && )
		binary_op( pf_or   , || )
		binary_op( pf_band , &  )
		binary_op( pf_bor  , |  )
		binary_op( pf_xor  , ^  )

	#undef binary_op
}

namespace {
	static petit_fex::var _1;
	static petit_fex::var _2;
	static petit_fex::var _3;
}

//
// example:
//   ・find_if( c.begin(), c.end(), 100<=X && X<=200 );
//   ・int r = (X*5+X/3+2) (6);
//


int main()
{
	using petit_fex::var;

	int r = ( _1 * 5 + _1 / 3 + 2 )(6);

	std::cout << r << std::endl;

	return 0;
}
