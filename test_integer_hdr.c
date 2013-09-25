#include <stdio.h>

#define N 20

/**
 * 浮動小数点バッファが使えない場合に精度を落とさず[0,8]までの輝度を再現する方法.
 * レンジ圧縮(1/8に輝度を圧縮)することで処理する.
 */
int main( void )
{
	int i, j;
	float radiance[N]; // 圧縮したい輝度
	int result[N][8]; // 圧縮された輝度(一つの輝度が8つに分解される)

	for ( i = 0; i < N; ++i )
	{
		radiance[i] = i;
	}

	// オフセットをつけて圧縮
	for ( i = 0; i < N; ++i )
	{
		for ( j = 0; j < 8; ++j )
		{
			result[i][j] = (int)( ( radiance[i] + j ) / 8 );
		}
	}

	// 復元(8つの係数を加算することで復元される)
	for ( i = 0; i < N; ++i )
	{
		int sum = 0;
		for ( j = 0; j < 8; ++j )
		{
			sum += result[i][j];
		}

		printf( "[" );
		for ( j = 0; j < 8; ++j )
		{
			printf( " %d", result[i][j] );
		}
		printf( " ]" );

		printf( " %d == %f\n", sum, radiance[i] );
	}

	return 0;
}
