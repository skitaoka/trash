#include "BackPropagation.hpp"

BackPropagation::BackPropagation( int input, int hidden, int output )
	: eta_( 0.3 )
	, alpha_( 0.0 )
	, input_( input )
	, hidden_( hidden )
	, output_( output )
	, hiddenLayer_( new double[ hidden + 1 ] )
	, outputLayer_( new double[ output ] )
	, wij_( new double*[ hidden + 1 ] )
	, wjk_( new double*[ output ] )
	, deltaj_( new double[ hidden + 1 ] )
	, deltak_( new double[ output ] )
	, deltaWji_( new double*[ hidden + 1 ] )
	, deltaWkj_( new double*[ output ] )
{
	hiddenLayer_[ hidden ] = 1.0;

	for ( int j = 0; j <= hidden; ++j )
	{
		wij_[j] = new double[ input + 1 ];
		for ( int i = 0; i <= input; ++i )
		{
			wij_[j][i] = (double)rand() / RAND_MAX - 0.5;
		}
	}

	for ( int k = 0; k < output; ++k )
	{
		wjk_[k] = new double[ hidden + 1 ];
		for ( int j = 0; j <= hidden; ++j )
		{
			wjk_[k][j] = (double)rand() / RAND_MAX - 0.5;
		}
	}

	for ( int j = 0; j <= hidden; ++j )
	{
		deltaWji_[j] = new double[ input + 1 ];
		for ( int i = 0; i <= input; ++i )
		{
			deltaWji_[j][i] = 0.0;
		}
	}

	for ( int k = 0; k < output; k++ )
	{
		deltaWkj_[k] = new double[ hidden + 1 ];
		for ( int j = 0; j <= hidden; ++j )
		{
			deltaWkj_[k][j] = 0.0;
		}
	}
}


BackPropagation::~BackPropagation()
{
	delete[] hiddenLayer_;
	delete[] outputLayer_;

	for ( int j = 0; j <= hidden_; ++j )
	{
		delete[] wij_[j];
	}
	delete[] wij_;

	for ( int k = 0; k < output_; ++k )
	{
		delete[] wjk_[k];
	}
	delete[] wjk_;


	delete[] deltaj_;
	delete[] deltak_;

	for ( int j = 0; j <= hidden_; ++j )
	{
		delete[] deltaWji_[j];
	}
	delete[] deltaWji_;

	for ( int k = 0; k < output_; ++k )
	{
		delete[] deltaWkj_[k];
	}
	delete[] deltaWkj_;
}


//! 入力を与えて結果を得る
void BackPropagation::foward( double* inputLayer )
{
	// 入力層->中間層
	for ( int j = 0; j <= hidden_; ++j )
	{
		double x = 0.0;

		for ( int i = 0; i < input_; ++i )
		{
			x += wij_[j][i] * inputLayer[i];
		}

		hiddenLayer_[j] = BackPropagation::sigmoid( x + wij_[j][input_] );
	}

	// 中間層->出力層
	for ( int k = 0; k < output_; ++k )
	{
		double x = 0.0;

		for ( int j = 0; j <= hidden_; ++j )
		{
			x += wjk_[k][j] * hiddenLayer_[j];
		}

		outputLayer_[k] = BackPropagation::sigmoid( x );
	}
}

void BackPropagation::back( double* inputLayer, double* teach )
{
	// 出力層->中間層
	for ( int k = 0; k < output_; ++k )
	{
		deltak_[k] = ( teach[k] - outputLayer_[k] ) *
				outputLayer_[k] * ( 1.0 - outputLayer_[k] );
	}

	for ( int k = 0; k < output_; ++k )
	{
		for ( int j = 0; j <= hidden_; ++j )
		{
			deltaWkj_[k][j] = eta_ * deltak_[k] * hiddenLayer_[j] + alpha_ * deltaWkj_[k][j];
			wjk_[k][j] += deltaWkj_[k][j];
		}
	}

	// 中間層->入力層
	for ( int j = 0; j <= hidden_; ++j )
	{
		double x = 0.0;

		for ( int k = 0; k < output_; ++k )
		{
			x += wjk_[k][j] * deltak_[k];
		}

		deltaj_[j] = x * hiddenLayer_[j] * ( 1.0 - hiddenLayer_[j] );
	}

	for ( int j = 0; j <= hidden_; ++j )
	{
		for ( int i = 0; i < input_; ++i )
		{
			deltaWji_[j][i] = eta_ * deltaj_[j] * inputLayer[i] + alpha_ * deltaWji_[j][i];
			wij_[j][i] += deltaWji_[j][i];
		}

		deltaWji_[j][input_] = eta_ * deltaj_[j] + alpha_ * deltaWji_[j][input_];
		wij_[j][input_] += deltaWji_[j][input_];
	}
}


//! 誤差を得る
double BackPropagation::getError( double* teach )
{
	double error = 0;

	for ( int k = 0; k < output_; ++k )
	{
		error += ( teach[k] - outputLayer_[k] ) *
				 ( teach[k] - outputLayer_[k] );
	}

	return error * 0.5;
}


void BackPropagation::debugOutput()
{
	for ( int k = 0; k < output_; ++k )
	{
		printf( "%lf ", outputLayer_[k] );
	}
	printf( "\n" );
}
