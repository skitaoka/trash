/*
 ピクセルの乗算テスト
*/
#if 1 // 速度テスト
#include <cstdio>
#include <ctime>

int main( void )
{
	std::size_t const N = 100000;
	clock_t times;
	int res = 0;

	times = std::clock();
	{
		for ( std::size_t n = 0; n < N; ++n )
		{
			for ( std::size_t i = 0; i < 256; ++i )
			{
				for ( std::size_t j = i; j < 256; ++j )
				{
					//res = static_cast<int>( ( i + 0.5 ) * ( j + 0.5 ) / 256.0 );
					res = ((i<<1)+1)*((j<<1)+1) >> 10;
				}
			}
		}
	}
	std::fprintf( stderr, "% 5.3lf [s] ... %d\n", ( std::clock() - times ) / double( CLOCKS_PER_SEC ), res );

	times = std::clock();
	{
		for ( std::size_t n = 0; n < N; ++n )
		{
			for ( std::size_t i = 0; i < 256; ++i )
			{
				for ( std::size_t j = i; j < 256; ++j )
				{
					res = ( (i*j) + ((i+j)>>1) ) >> 8;
				}
			}
		}
	}
	std::fprintf( stderr, "% 5.3lf [s] ... %d\n", ( std::clock() - times ) / double( CLOCKS_PER_SEC ), res );

	return 0;
}

#else // 精度テスト

#include <iostream>

int main( void )
{
	std::size_t count = 0;
	for ( std::size_t i = 0; i < 256; ++i )
	{
		for ( std::size_t j = i; j < 256; ++j )
		{
			// シンプルな実装
			int a = static_cast<int>( ( i + 0.5 ) * ( j + 0.5 ) / 256.0 );

			// 高速化
			/* 式変形
			( ( i + 0.5 ) / 256.0 ) * ( ( j + 0.5 ) / 256.0 ) * 256.0
			( i + 0.5 ) * ( j + 0.5 ) / 256.0
			( i*j + i/2 + j/2 + 1/4 ) / 256
			( i*j + i>>1 + j>>1 + 1>>2 ) >> 8
			( i*j<<2 + i<<1 + j<<1 + 1 ) >> 10
			( (i*j<<2) + (i+j<<1) + 1 ) >> 10
			*/

			/* 素直な実装
			int b = (i*j<<2) + (i+j<<1) + 1;
			if ( b < 0 ) b = 0;
			else b >>= 10;
			*/

			/* ビット演算でクランプ
			int b = (i*j<<2) + (i+j<<1) + 1;
			b = ( ( ~b >> 31 ) & b ) >> 10;
			*/

			/* 足し算を削る
			int b = (i*j<<1) + (i+j);
			b = ( ( ~b >> 31 ) & b ) >> 9;
			*/

			/* もうちょっと変形
			int b = (i*j) + (i+j>>1);
			b = ( ( ~b >> 31 ) & b ) >> 8;
			*/

			int b = ( (i*j) + (i+j>>1) ) >> 8;
			//int b = ((i<<1)+1)*((j<<1)+1) >> 10;

			if ( a == b )
			{
			}
			else
			{
				++count;
			}
		}
	}
	std::cerr << count << std::endl;

	return 0;
}
#endif
