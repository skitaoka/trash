/**
 * 1..nの総和S_nを求める
 */
int S_n( int n )
{
	return n * ( n + 1 ) / 2;
}

/**
 * 等差数列a_n=a+(n-1)dの総和S_nを求める
 *
 * a 初項
 * d 公差
 */
template <typename T>
T S_n_arithmetic_series( int n, T a, T d )
{
	return n * ( 2 * a + ( n - 1 ) * d ) / 2;
}

/**
 * 等比数列a_n=a*r^(n-1)の総和S_nを求める
 *
 * a 初項
 * r 公比
 */
template <typename T>
T S_n_geometric_series( int n, T a, T r )
{
	if ( r == 1 )
	{
		return n * a;
	}
	else
	{
		T prod( 1 );
		for ( int i = 0; i < n; ++i )
		{
			prod *= r;
		}
		return a * ( 1 - prod ) / ( 1 - r );
	}
}
