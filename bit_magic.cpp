#if !defined( BIT_MAGIC_HPP )
#define BIT_MAGIC_HPP

/**
 * 32bit整数のabsを求める．
 * 正数の場合
 * x            , 0XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
 * y            , 00000000000000000000000000000000
 * x ^ y        , 0XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
 * ( x ^ y ) - y, 変化なし
 *
 * 負数の場合
 * x            , 1XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
 * y            , 11111111111111111111111111111111
 * x ^ y        , 0xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
 * ( x ^ y ) - y, ~x + 1 と同じになる
 */
inline int abs( int x )
{
	int y = x >> 31;
	return ( x ^ y ) - y;
}

/**
 * 32bit整数のminを求める．
 * x >= y の場合
 * x            , XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
 * y            , YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY
 * x - y        , 0ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ
 * z            , 00000000000000000000000000000000
 * x & z        , 00000000000000000000000000000000
 * x & ~z       , YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY
 *
 * x < y 場合
 * x            , XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
 * y            , YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY
 * x - y        , 1ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ
 * z            , 11111111111111111111111111111111
 * x & ~z       , XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
 * x & z        , 00000000000000000000000000000000
 */
inline int min( int x, int y )
{
	int z = ( x - y ) >> 31;
	// return ( x & z ) | ( y & ~z );
	return ( ( x ^ y ) & z ) ^ y;
}

/**
 * 32bit整数のmaxを求める．
 * x >= y の場合
 * x            , XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
 * y            , YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY
 * x - y        , 0ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ
 * z            , 00000000000000000000000000000000
 * x & ~z       , XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
 * x & z        , 00000000000000000000000000000000
 *
 * x < y 場合
 * x            , XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
 * y            , YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY
 * x - y        , 1ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ
 * z            , 11111111111111111111111111111111
 * x & ~z       , 00000000000000000000000000000000
 * x & z        , YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY
 */
inline int max( int x, int y )
{
	int z = ( x - y ) >> 31;
//	return ( x & ~z ) | ( y & z );
	return ( ( x ^ y ) & z ) ^ x;
}

inline float abs( float x )
{
	*((int*)&x) &= 0x7FFFFFFF;
	return x;
}


/**
 * 8bpp抜き色転送
 *
 *
 */
void copy8bpp( unsigned int* src, unsigned int* dst, unsigned int len, unsigned int key )
{
	unsigned int k = key | ( key << 8 ) | ( key << 16 ) | ( key << 24 );

	len >>= 2;

	for ( int i = 0; i < len; ++i ) {
		unsigned int s = *src++;
		unsigned int m = s ^ k;
		m = m | ( ( m & 0x7f7f7f7f ) + 0x7f7f7f7f );
		m = ( m & 0x80808080 ) >> 7;
		m = ( m + 0x7f7f7f7f ) ^ 0x7f7f7f7f;
		// *dst = ( s & m ) | ( *dst & ~m );
		*dst = ( ( s ^ *dst ) & m ) ^ *dst;
	}
}


/**
 * 立ってるビットの数
 */
inline int count_bits( int bits )
{
	// count が常に正なら最初の行は
	// count = count - ( ( count & 0xAAAAAAAA ) >> 1 );
	// と書ける.
	bits = ( bits & 0x55555555 ) + ( ( bits >>  1 ) & 0x55555555 );
	bits = ( bits & 0x33333333 ) + ( ( bits >>  2 ) & 0x33333333 );
	bits = ( bits & 0x0f0f0f0f ) + ( ( bits >>  4 ) & 0x0f0f0f0f );
	bits = ( bits & 0x00ff00ff ) + ( ( bits >>  8 ) & 0x00ff00ff );
	return ( bits & 0x0000ffff ) + ( ( bits >> 16 ) & 0x0000ffff );
}

/**
 * 逆順にビットを並べ替える
 */
inline int reverse_bits( int bits )
{
	bits = ( ( bits & 0x0000ffff ) << 16 ) | ( ( bits >> 16 ) & 0x0000ffff );
	bits = ( ( bits & 0x00ff00ff ) <<  8 ) | ( ( bits >>  8 ) & 0x00ff00ff );
	bits = ( ( bits & 0x0f0f0f0f ) <<  4 ) | ( ( bits >>  4 ) & 0x0f0f0f0f );
	bits = ( ( bits & 0x33333333 ) <<  2 ) | ( ( bits >>  2 ) & 0x33333333 );
	return ( ( bits & 0x55555555 ) <<  1 ) | ( ( bits >>  1 ) & 0x55555555 );
}

/**
 * tzc; training zero count.
 * 右から数えてゼロが並んでいる数
 */
inline int tzc( int x )
{
	return count_bits( ~x & ( x - 1 ) );
}

/**
 * lzc; leading zero count
 * 左から数えてゼロが並んでいる数
 */
inline int lzc( int x )
{
	int y, n = 32;
	y = x >> 16; if ( y ) { n = n - 16; x = y; }
	y = x >>  8; if ( y ) { n = n -  8; x = y; }
	y = x >>  4; if ( y ) { n = n -  4; x = y; }
	y = x >>  2; if ( y ) { n = n -  2; x = y; }
	y = x >>  1; if ( y ) { return n - 2; } // n = n - 1; x = y; return n - x;
	return n - x;
	// 通常xはこの時点で1になっているのでn-1で良いのだが，
	// lzc(0)の時32を返すようにするためこのようになっている.
}


#if !defined( NDEBUG )

#include <cstdio>
#include <ctime>

int lzc_count( int x )
{
  x = x | ( x >>  1 );
  x = x | ( x >>  2 );
  x = x | ( x >>  4 );
  x = x | ( x >>  8 );
  x = x | ( x >> 16 );
  return count_bits( ~x );
}

inline int lzc_nonif( int x )
{
	int y, m, n;

	y = - ( x >> 16 );
	m = ( y >> 16 ) & 16;
	n = 16 - m;
	x = x >> m;

	y = x - 0x100;
	m = ( y >> 16 ) & 8;
	n = n + m;
	x = x << m;

	y = x - 0x1000;
	m = ( y >> 16 ) & 4;
	n = n + m;
	x = x << m;

	y = x - 0x4000;
	m = ( y >> 16 ) & 2;
	n = n + m;
	x = x << m;

	y = x >> 14;
	m = y & ~( y >> 1 );

	return n + 2 - m;
}

inline int lzc_double( int x )
{
	union {
		__int64 as_long;
//		long long as_long; // for gcc
		double as_double;
	} data;

	data.as_double = static_cast<double>( x + 0.5 );
	return 31 + 1023 - ( data.as_long >> 52 );
}

inline int lzc_reverse( int x )
{
	return tzc( reverse_bits( x ) );
}

inline int lzc_debug( int x )
{
	for ( int n = 31; n >= 0; --n ) {
		if ( x & ( 1 << n ) ) {
			return 31 - n;
		}
	}
	return 32;
}

int main( int argc, char* argv[] )
{
	const int s = 0x00000000;
	const int n = 0x10000000;
	int times;
	int x, a;

	times = std::clock();
	x = 0;
	for ( int i = s; i < n; ++i ) {
		x += lzc_debug( i );
	}
	times = std::clock() - times;
	std::printf( "% 5.3lf [s] ... done. 0x%0x debug\n", (double)times / CLOCKS_PER_SEC, x );

	times = std::clock();
	x = 0;
	for ( int i = s; i < n; ++i ) {
		x += lzc( i );
	}
	times = std::clock() - times;
	std::printf( "% 5.3lf [s] ... done. 0x%0x if\n", (double)times / CLOCKS_PER_SEC, x );

	times = std::clock();
	x = 0;
	for ( int i = s; i < n; ++i ) {
		x += lzc_count( i );
	}
	times = std::clock() - times;
	std::printf( "% 5.3lf [s] ... done. 0x%0x count\n", (double)times / CLOCKS_PER_SEC, x );

	times = std::clock();
	x = 0;
	for ( int i = s; i < n; ++i ) {
		x += lzc_nonif( i );
	}
	times = std::clock() - times;
	std::printf( "% 5.3lf [s] ... done. 0x%0x nonif\n", (double)times / CLOCKS_PER_SEC, x );

	times = std::clock();
	x = 0;
	for ( int i = s; i < n; ++i ) {
		x += lzc_double( i );
	}
	times = std::clock() - times;
	std::printf( "% 5.3lf [s] ... done. 0x%0x double\n", (double)times / CLOCKS_PER_SEC, x );

	times = std::clock();
	x = 0;
	for ( int i = s; i < n; ++i ) {
		x += lzc_reverse( i );
	}
	times = std::clock() - times;
	std::printf( "% 5.3lf [s] ... done. 0x%0x reverse\n", (double)times / CLOCKS_PER_SEC, x );

	return 0;
}

#endif

#endif // BIT_MAGIC_HPP
