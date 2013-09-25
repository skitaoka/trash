// 隠れマルコフモデル(hidden Markov model; HMM)
public class HMM
{
	// 状態の有限集合Q={q_{1},...,q_{N}}
	private String[] state = { "A", "B", "C" };

	// 出力記号の有限集合sigma={o_{1},...,o_{M}}
	private int[] sigma = { 0, 1, 2, 3 };

	// 状態遷移確率分布A={a_{i}_{j}}
	private double[][] a =
	{	// "A"  "B"  "C"
		{ 0.2, 0.3, 0.5, }, // "A"
		{ 0.3, 0.2, 0.5, }, // "B"
		{ 0.2, 0.3, 0.5, }, // "C"
	};

	// 記号出力確率分布B={b_{i}(o_{t})}
	private double[][] b =
	{	// 0    1    2    3
		{ 0.4, 0.2, 0.2, 0.2 }, // "A"
		{ 0.2, 0.4, 0.2, 0.2 }, // "B"
		{ 0.2, 0.2, 0.4, 0.2 }, // "C"
	};

	// 初期状態確率分布pi={pi_{i}}
	private double[] pi =
	{
		0.8, // "A"
		0.1, // "B"
		0.1, // "C"
	};

	/*
	 * 評価問題 (evaluation problem)
	 */

	// 前向きアルゴリズム(forward algorithm)
	public double forwardAlgorithm( int[] o )
	{
		// 1.各状態i=1,...,Nに対して、前向き確率を初期化する。
		// alpha_{1}(i)=pi_{i} b_{i}(o_{1})
		double[][] alpha = new double[ o.length ][ state.length ];
		for ( int i = 0; i < state.length; ++i )
		{
			alpha[0][i] = pi[i] * b[i][ o[0] ];
		}

		// 2.各時刻t=1,...,T-1各状態j=1,...,Nについて、前向き確率を再帰的に計算する。
		// alpha_{t+1}(j)=[sigma_{i=1}^{N} alpha_{t} a_{i}_{j}] b_{j}(o_{t+1})
		for ( int t = 1; t < o.length; ++t )
		{
			for ( int j = 0; j < alpha[0].length; ++j )
			{
				double temp = 0;
				for ( int i = 0; i < alpha[0].length; ++i )
				{
					temp += alpha[ t - 1 ][i] * a[i][j];
				}
				alpha[t][j] = temp * b[j][ o[t] ];
			}
		}

		// 3.最終確率の計算。
		// P(o_{1}^{T}|M)=sigma_{i=1}^{N} alpha_{T}(i)
		double forwardProbability = 0;
		for ( int i = 0; i < alpha[0].length; ++i )
		{
			forwardProbability += alpha[ alpha.length - 1 ][i];
		}
		return forwardProbability;
	}

	// 後ろ向きアルゴリズム(backward algorithm)
	public double backwardAlgorithm( int[] o )
	{
		// 1.各状態i=1,...,Nに対して、後向き確率を初期化する。
		// beta_{T}=1
		double[][] beta = new double[ o.length ][ state.length ];
		for ( int i = 0; i < state.length; ++i )
		{
			beta[ beta.length - 1 ][i] = 1;
		}

		// 2.各時刻t=T-1,...,1各状態i=1,...,Nについて、後向き確率を再帰的に計算する。
		// beta_{t}(i)=sigma_{j=1}^{N} a_{i}_{j} b_{j}(o_{t+1}) beta_{t+1}(j)
		for ( int t = o.length - 1; t >= 1; --t )
		{
			for ( int i = 0; i < beta[0].length; ++i )
			{
				beta[ t - 1 ][i] = 0;
				for ( int j = 0; j < beta[0].length; ++j )
				{
					beta[ t - 1 ][i] += a[i][j] * b[j][ o[t] ] * beta[t][j];
				}
			}
		}

		// 3.最終確率の計算。
		// P(o_{1}^{T}|M)=sigma_{i=1}^{N} pi_{i} b_{i}(o_{1}) beta_{1}(i)
		double backwardProbability = 0;
		for ( int i = 0; i < beta[0].length; i++ )
		{
			backwardProbability += pi[i] * b[i][ o[0] ] * beta[0][i];
		}
		return backwardProbability;
	}

	/*
	 * 復号化問題(decoding problem)
	 */
	// ビタビ・アルゴリズム(Viterbi algorithm)
	public double viterbiAlgorithm( int[] o ) {
		// 1.各状態i=1,...,Nに対して、変数の初期化を行う。
		// delta_{1}(i)=pi_{i} b_{i}(o_{1})
		// psi_{1}(i)=0
		double[][] delta = new double[ o.length ][ state.length ];
		int[][] psi = new int[ o.length ][ state.length ];
		for ( int i = 0; i < state.length; ++i )
		{
			delta[0][i] = pi[i] * b[i][ o[0] ];
			psi[0][i] = 0;
		}

		// 2.各時刻t=1,...,T-1、各状態j=1,...,Nについて、再起計算を実行。
		// delta_{t+1}=max_{i}[delta_{t}(i) a_{i}_{j}] b_{j}(o_{t+1})
		// psi_{t+1}=argmax_{i}[delta_{t}(i) a_{i}_{j}]
		for ( int t = 1; t < o.length; ++t )
		{
			for ( int j = 0; j < delta[0].length; ++j )
			{
				double maxDeltaA = 0;
				int argmaxDeltaA = 0;
				for ( int i = 0; i < delta[0].length; i++ )
				{
					double deltaA = delta[ t - 1 ][i] * a[i][j];
					if ( maxDeltaA < deltaA )
					{
						maxDeltaA = deltaA;
						argmaxDeltaA = i;
					}
				}
				delta[t][j] = maxDeltaA * b[j][ o[t] ];
				psi[t][j] = argmaxDeltaA;
			}
		}

		// 3.再起計算の終了。
		// P^{^}=max_{i} delta_{T}(i)
		// q_{T}^{^}=argmax_{i} delta_{T}(i)
		double maxDelta = 0;
		int argmaxDelta = 0;
		for ( int i = 0; i < delta[0].length; i++ )
		{
			double tempDelta = delta[ delta.length - 1 ][i];
			if ( maxDelta < tempDelta )
			{
				maxDelta = tempDelta;
				argmaxDelta = i;
			}
		}

		// 4.バックトラックによる最適状態遷移系列の復元。
		// t=T-1,...,1に対して、次を実行する。
		// q_{t}^{^}=psi_{t+1}(q_{t}^{^}+1)
		int[] q = new int[ o.length ];

		q[ o.length - 1 ] = argmaxDelta;
		for ( int t = o.length - 1; t >= 1; --t )
		{
			q[ t - 1 ] = psi[ t ][ q[ t ] ];
		}

		// 最適な系列を出力
		for ( int x : q )
		{
			System.out.print( state[x] );
		}
		System.out.println();

		return maxDelta;
	}

	/*
	 * 推定化問題(estimation problem)
	 */

	// 前向き・後ろ向きアルゴリズム(forward-backward algorithm)
	// バウム・ウェルチのアルゴリズム(Baum-Welch algorithm)
	public void forwardBackwardAlgorithm()
	{
	}

	public static void main( String[] args )
	{
		int[] o = { 0, 0, 0, };
		for ( int x : o )
		{
			System.out.print( x );
		}
		System.out.println();

		HMM hmm = new HMM();
		System.out.println( hmm.forwardAlgorithm( o ) );
		System.out.println( hmm.backwardAlgorithm( o ) );
		System.out.println( hmm.viterbiAlgorithm( o ) );
	}
}
