#include <cmath>
#include <iostream>
#include <complex>

/**
 * 離散フーリエ変換(DFT; )
 */
void DFT( std::size_t n, std::complex<double> const in[], std::complex<double> out[] )
{
	std::complex<double> w[n];
	for ( std::size_t i = 0; i < n; ++i )
	{
		double t = 2 * M_PI * i / n;
		w[i] = std::complex<double>( std::cos( t ), -std::sin( t ) );
	}

	for ( std::size_t i = 0; i < n; ++i )
	{
		out[i] = std::complex<double>();
		for ( std::size_t j = 0; j < n; ++j )
		{
			out[i] += in[j] * w[(i*j)%n];
		}
	}
}

/**
 * 離散逆フーリエ変換(Inverse DFT)
 */
void IDFT( std::size_t n, std::complex<double> const in[], std::complex<double> out[] )
{
	std::complex<double> w[n];
	for ( std::size_t i = 0; i < n; ++i )
	{
		double t = 2 * M_PI * i / n;
		w[i] = std::complex<double>( std::cos( t ),  std::sin( t ) );
	}

	for ( std::size_t i = 0; i < n; ++i )
	{
		out[i] = std::complex<double>();
		for ( std::size_t j = 0; j < n; ++j )
		{
			out[i] += in[j] * w[(i*j)%n];
		}
	}
}

int main( void )
{
	std::size_t const n = 6;
	std::complex<double> a[n] =
	{
		std::complex<double>( 1 ),
		std::complex<double>( 2 ),
		std::complex<double>( 1 ),
		std::complex<double>( 3 ),
		std::complex<double>( 2 ),
		std::complex<double>( 4 ),
	};
	std::complex<double> b[n];
	std::complex<double> c[n];

	std::cout << "a" << std::endl;
	for ( std::size_t i = 0; i < n; ++i )
	{
		std::cout << a[i] << std::endl;
	}

	DFT( n, a, b );

	std::cout << "b" << std::endl;
	for ( std::size_t i = 0; i < n; ++i )
	{
		std::cout << b[i] << std::endl;
	}

	IDFT( n, b, c );

	std::cout << "c" << std::endl;
	for ( std::size_t i = 0; i < n; ++i )
	{
		std::cout << c[i] << std::endl;
	}

	return 0;
}
