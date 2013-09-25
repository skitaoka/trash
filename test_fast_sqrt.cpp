#include <cstdio>
#include <cmath>
#include <ctime>

/* ニュートン法による逆数の漸化式 1 / a
 * x_{n+1} = 2 * x_n - a * x_n^2
 *         = x_n * ( 1 + h_n + h_n^2 ), h_n = 1 - a * x_n
 *         = x_n * ( 1 + h_n ) * ( 1 + h_n^2 )
 *         = x_n * ( 1 + ( 1 + h_n^2 ) + ( h_n + h_n^2 ) )
 */

/* ニュートン法による平方根の漸化式 \sqrt{a}
 * x_{n+1} = 0.5 * ( x_n + a / x_n )
 */

/* ニュートン法による平方根の逆数の漸化式 1 / \sqrt{a}
 * x_{n+1} = x_n * ( 1.5 - 0.5 * a * x_n^2 )
 *         = x_n * ( 1.0 + h_n * ( 0.5 + 0.375 * h_n ) ), h_n = 1.0 - a * x_n^2
 *         = x_n * ( 1.0 + h_n * ( 0.5 + h_n * ( 0.375 + 0.3125 * h_n ) ) )
 *         = x_n * ( 1.0 + h_n * ( 0.5 + h_n * ( 0.375 + h_n * ( 0.3125 + h_n * ( 0.2734375 + h_n * 0.24609375 ) ) ) ) )
 */

namespace orz {

	template < typename T >
	T fast_inv_sqrt( T a );

	/*
	 * 精度 6 桁程度
	 */
	template <>
	float fast_inv_sqrt( float a )
	{
		unsigned long int b = *reinterpret_cast< unsigned long int* >( &a );

		if ( b & 0x80000000UL ) {
			std::fprintf( std::stderr, "domain error\n" );
			return 0;
		}

		unsigned long int e = ( ( 127UL + 63UL ) << 23UL ) - ( ( b >> 1 ) & 0x7F800000UL );
		float x_n = *reinterpret_cast< float* >( &e );
		float h_n;

		h_n = 1.0f - a * x_n * x_n;
		x_n += x_n * h_n * ( ( 1.0f / 2.0f ) + h_n * ( ( 3.0f / 8.0f ) + h_n * (
			( 5.0f / 16.0f ) + h_n * ( ( 35.0f / 128.0f ) + h_n * ( 63.0f / 256.0f ) ) ) ) );
		h_n = 1.0f - a * x_n * x_n;
		x_n += x_n * h_n * ( ( 1.0f / 2.0f ) + h_n * ( ( 3.0f / 8.0f ) + h_n * (
			( 5.0f / 16.0f ) + h_n * ( ( 35.0f / 128.0f ) + h_n * ( 63.0f / 256.0f ) ) ) ) );
		h_n = 1.0f - a * x_n * x_n;
		x_n += x_n * h_n * ( ( 1.0f / 2.0f ) + h_n * ( ( 3.0f / 8.0f ) + h_n * (
			( 5.0f / 16.0f ) + h_n * ( ( 35.0f / 128.0f ) + h_n * ( 63.0f / 256.0f ) ) ) ) );

		return x_n;
	}

	/*
	 * 精度 14 桁程度
	 */
	template <>
	double fast_inv_sqrt( double a )
	{
		unsigned long long int b = *reinterpret_cast<unsigned long long int *>( &a );

		if ( b & 0x8000000000000000ULL ) {
			fprintf( stderr, "domain error\n" );
			return 0;
		}

		unsigned long long int e = ( ( 1023ULL + 511ULL ) << 52ULL ) - ( ( b >> 1 ) & 0x7FF0000000000000ULL );
		double x_n = *reinterpret_cast< double* >( &e );
		double h_n;

		h_n = 1.0 - a * x_n * x_n;
		x_n += x_n * h_n * ( ( 1.0 / 2.0 ) + h_n * ( ( 3.0 / 8.0 ) + h_n * (
			( 5.0 / 16.0 ) + h_n * ( ( 35.0 / 128.0 ) + h_n * ( 63.0 / 256.0 ) ) ) ) );
		h_n = 1.0 - a * x_n * x_n;
		x_n += x_n * h_n * ( ( 1.0 / 2.0 ) + h_n * ( ( 3.0 / 8.0 ) + h_n * (
			( 5.0 / 16.0 ) + h_n * ( ( 35.0 / 128.0 ) + h_n * ( 63.0 / 256.0 ) ) ) ) );
		h_n = 1.0 - a * x_n * x_n;
		x_n += x_n * h_n * ( ( 1.0 / 2.0 ) + h_n * ( ( 3.0 / 8.0 ) + h_n * (
			( 5.0 / 16.0 ) + h_n * ( ( 35.0 / 128.0 ) + h_n * ( 63.0 / 256.0 ) ) ) ) );

		return x_n;
	}
}

/**
 * -O3で標準の1.3倍程度
 * -ffast-mathをつけるとstdの方が爆速
 *
 * -march=i686 -DNDEBUG -O3 -ffast-math -msse3 -mfpmath=sse -fomit-frame-pointer -funroll-loops
 */
int main( void )
{
	std::printf( "error check.\n" );
		std::printf( "x\t\tstd\t\torz\t\terror\n" );
	for ( int i = 0; i < 100; ++i ) {
		double x = i;
		double t = std::sqrt( x );
		double f = x * orz::fast_inv_sqrt( x );
		double e = std::fabs( t - f );
		std::printf( "%lf\t%lf\t%lf\t%22.20lf\n", x, t, f, e );
	}

	std::printf( "speed check.\n" );

	const unsigned long int n = 10000000;
	int times;

	std::printf( "orz:" );
	times = clock();
	for ( std::size_t i = 0; i < n; ++i ) {
		double x = i;
		x *= orz::fast_inv_sqrt( x );
	}
	times = clock() - times;
	std::printf( "% 5.3lf [s] ... done.\n", (double)times / CLOCKS_PER_SEC );

	std::printf( "std:" );
	times = clock();
	for ( std::size_t i = 0; i < n; ++i ) {
		double x = i;
		x = std::sqrt( x );
	}
	times = clock() - times;
	std::printf( "% 5.3lf [s] ... done.\n", (double)times / CLOCKS_PER_SEC );

	return 0;
}
