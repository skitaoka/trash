
public class EM {

	public static void main( String[] args )
	{
		int[] data =
		{
			1, 0, 0, 0, 1, 0, 0, 0, 1, 1,
			0, 1, 1, 1, 1, 0, 0, 0, 1, 0,
			0, 1, 0, 1, 1, 0, 1, 1, 1, 0,
			0, 0, 1, 1, 0, 1, 1, 1, 0, 0,
		};

		// ラムダ
		double lamda = 0.6;
		for ( int i = 0; i < 10000; ++i )
		{
			double temp = lamda( data, lamda );
			if ( Math.abs( lamda - temp ) < 1e-3 )
			{
				System.out.println( i );
				break;
			}
			lamda = temp;
		}
		System.out.println( "ラムダ：" + lamda );

		// 近似値
		double falseValue = getProb( 1, lamda );
		System.out.println( "近似値：" + falseValue );

		// 真値
		double trueValue = 0.0;
		for ( int x : data )
		{
			trueValue += x;
		}
		System.out.println( "真値　：" + ( trueValue / data.length ) );
	}

	/**
	 * パラメータλの新しい値
	 */
	public static double lamda( int[] data, double oldLamda )
	{
		double newLamda = 0.0;
		for ( int x : data )
		{
			newLamda += expecte_lamda( x, oldLamda );
		}
		return newLamda / data.length;
	}

	/**
	 * 期待値の計算
	 */
	public static double expecte_lamda( int data, double lamda )
	{
		return lamda * getProbA( data ) / getProb( data, lamda );
	}

	/**
	 * 混合モデルにおけるdataの生起確率
	 */
	public static double getProb( int data, double lamda )
	{
		return lamda * getProbA( data ) + ( 1.0 - lamda ) * getProbB( data );
	}

	/**
	 * モデルAにおけるdataの生起確率
	 */
	public static double getProbA( int data )
	{
		if ( data == 1 ) {
			return 0.4;
		} else {
			return 0.6;
		}
	}

	/**
	 * モデルBにおけるdataの生起確率
	 */
	public static double getProbB( inte data )
	{
		if ( data == 1 ) {
			return 0.5;
		} else {
			return 0.5;
		}
	}
}
