#if !defined( BACKPROPAGATION_HPP )
#define BACKPROPAGATION_HPP

#include <cstdio>
#include <cstdlib>
#include <cmath>

#define for if ( 0 ) ; else for

/**
 * ニューラルネットワーク
 * 誤差逆伝播法(バックプロパゲーション)
 */
class BackPropagation
{
public:
	BackPropagation( int input, int hidden, int output );
	~BackPropagation();

	//! 入力を与えて結果を得る
	void foward( double* inputLayer );

	//! 教師データを与えて学習
	void back( double* inputLayer, double* teach );

	//! 誤差を得る
	double getError( double* teach );

	void debugOutput();

private:
	double eta_; //!< 学習係数
	double alpha_; //!< 前回の変化量から学習量をコントロールするためのもの

	int input_; //!< 入力層数
	int hidden_; //!< 中間層数
	int output_; //!< 出力層数

	double* hiddenLayer_; //!< 中間層
	double* outputLayer_; //!< 出力層
	double** wij_; //!< 荷重（入力層->中間層）
	double** wjk_; //!< 荷重（中間層->出力層）

	double* deltaj_; //!< 誤差（入力層->中間層）
	double* deltak_; //!< 誤差（中間層->出力層）
	double** deltaWji_; //!< 荷重修正値（入力層->中間層）
	double** deltaWkj_; //!< 荷重修正値（入力層->中間層）

private:
	//! シグモイド関数
	inline static double sigmoid( double x )
	{
		return 1.0 / ( 1.0 + exp( -x ) );
	}
};

#endif // !defined( BACKPROPAGATION_HPP )
