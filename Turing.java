import java.awt.Color;
import java.awt.Graphics;
import java.awt.Image;
import java.awt.image.BufferedImage;

/**
 * 反応拡散系とは
 *   dU/dt = f(U,V) + ⊿U
 *   dV/dt = g(U,V) + ⊿V
 * という形をした非線形の連立偏微分方程式
 *
 * Gray-Scottモデル
 *   dU/dt =  U^2 * V - ( F + k ) * (     U ) + ⊿U
 *   dV/dt = -U^2 * V + ( F     ) * ( 1 - V ) + ⊿V
 *
 * Turing Pattern
 *   2成分(U, V)からなる反応ダイナミクス
 *   U: 自分自身の生成を触媒・Vの生成を触媒
 *   V: 自然に分解・Uを抑制
 *   例)
 *     dU/dt = f(U,V) = 5 * U - 6 * V + 1
 *     dV/dt = g(U,V) = 6 * U - 7 * V + 1
 *
 * 拡散による相互作用(例)
 *   dU[i]/dt = f(U[i],V[i]) + D_U * (U[i+1] - U[i]) + D_U * (U[i-1] - U[i])
 *   dV[i]/dt = g(U[i],V[i]) + D_V * (V[i+1] - V[i]) + D_V * (V[i-1] - V[i])
 *   D_U, D_V: 拡散係数
 */

/**
 * <applet code="Turing" width="1024" height="600" alt="Turing pattern">
 * </applet>
 */
public final class Turing extends java.applet.Applet implements Runnable
{
	private static final int LOG2N = 9;
	private static final int N = 1 << LOG2N;
	private static final int L = 256;
	private static final int[] color;

	private static final double D_T = 0.025;
	private static final double D_U = 2.0;
	private static final double D_V = 5.2;

	static
	{
		color = new int[L];
		for ( int i = 0; i < L; ++i )
		{
			color[i] = Color.getHSBColor( i / (float)L, 0.8f, 1.0f ).getRGB();
		}
	}

	private final double[][] U_ = new double[2][ N * N ];
	private BufferedImage imageU_;

	private final double[][] V_ = new double[2][ N * N ];
	private BufferedImage imageV_;

	// ここでのdouble bufferingは冗長
	private Image backbuffer_;
	private Graphics graphicsContext_;
	private volatile Thread kicker_;

	@Override
	public void init()
	{
		imageU_ = new BufferedImage( N, N, BufferedImage.TYPE_INT_RGB );
		imageV_ = new BufferedImage( N, N, BufferedImage.TYPE_INT_RGB );

		backbuffer_ = createImage( getWidth(), getHeight() );
		graphicsContext_ = backbuffer_.getGraphics();

		// 初期ノイズを加える
		for ( int i = 0, it = 0; i < N; ++i )
		{
			for ( int j = 0; j < N; ++j, ++it )
			{
				U_[0][it] = 1 + ( 2 * Math.random() - 1 ) * 1e-3;
				V_[0][it] = 1 + ( 2 * Math.random() - 1 ) * 1e-3;
			}
		}
	}

	@Override
	public void start()
	{
		if ( null == kicker_ )
		{
			kicker_ = new Thread( this );
			kicker_.start();
		}
	}

	@Override
	public void update( Graphics g )
	{
		paint( g );
	}

	@Override
	public void paint( Graphics g )
	{
		g.drawImage( backbuffer_, 0, 0, null );
	}

	public void run()
	{
		try
		{
			Thread thisThread = Thread.currentThread();
			thisThread.setPriority( Thread.NORM_PRIORITY - 1 );
			while ( kicker_ == thisThread )
			{
				advanceFrame();
				renderScene( graphicsContext_ );
				repaint();
				Thread.sleep( 20L );
			}
		}
		catch ( InterruptedException e )
		{
		}
		finally
		{
			kicker_ = null;
		}
	}

	private void advanceFrame()
	{
		// シミュレーション
		double[] Uprev = U_[0];
		double[] Unext = U_[1];
		double[] Vprev = V_[0];
		double[] Vnext = V_[1];
		for ( int i = 0, it = 0; i < N; ++i )
		{
			int im1 = ( i - 1 ) & ( N - 1 );
			int ip1 = ( i + 1 ) & ( N - 1 );

			for ( int j = 0; j < N; ++j, ++it )
			{
				int jm1 = ( j - 1 ) & ( N - 1 );
				int jp1 = ( j + 1 ) & ( N - 1 );

				double V_U = 
						+ 5 * Uprev[it] - 6 * Vprev[it] + 1
						+ D_U * ( Uprev[(im1<<LOG2N)+jm1] - Uprev[it] )
						+ D_U * ( Uprev[(im1<<LOG2N)+j  ] - Uprev[it] )
						+ D_U * ( Uprev[(im1<<LOG2N)+jp1] - Uprev[it] )
						+ D_U * ( Uprev[(i  <<LOG2N)+jm1] - Uprev[it] )
						+ D_U * ( Uprev[(i  <<LOG2N)+jp1] - Uprev[it] )
						+ D_U * ( Uprev[(ip1<<LOG2N)+jm1] - Uprev[it] )
						+ D_U * ( Uprev[(ip1<<LOG2N)+j  ] - Uprev[it] )
						+ D_U * ( Uprev[(ip1<<LOG2N)+jp1] - Uprev[it] );

				double V_V = 
						+ 6 * Uprev[it] - 7 * Vprev[it] + 1
						+ D_V * ( Vprev[(im1<<LOG2N)+jm1] - Vprev[it] )
						+ D_V * ( Vprev[(im1<<LOG2N)+j  ] - Vprev[it] )
						+ D_V * ( Vprev[(im1<<LOG2N)+jp1] - Vprev[it] )
						+ D_V * ( Vprev[(i  <<LOG2N)+jm1] - Vprev[it] )
						+ D_V * ( Vprev[(i  <<LOG2N)+jp1] - Vprev[it] )
						+ D_V * ( Vprev[(ip1<<LOG2N)+jm1] - Vprev[it] )
						+ D_V * ( Vprev[(ip1<<LOG2N)+j  ] - Vprev[it] )
						+ D_V * ( Vprev[(ip1<<LOG2N)+jp1] - Vprev[it] );

				Unext[it] = Uprev[it] + V_U * D_T;
				Vnext[it] = Vprev[it] + V_V * D_T;
			}
		}
		U_[0] = Unext;
		U_[1] = Uprev;
		V_[0] = Vnext;
		V_[1] = Vprev;
	}

	private void renderScene( Graphics g )
	{
		// 可視化用のデータを生成
		double[] U = U_[0];
		double[] V = V_[0];

		double minU = Double.POSITIVE_INFINITY;
		double maxU = Double.NEGATIVE_INFINITY;
		double minV = Double.POSITIVE_INFINITY;
		double maxV = Double.NEGATIVE_INFINITY;
		for ( int i = 0; i < N * N; ++i )
		{
			if ( minU > U[i] ) minU = U[i];
			if ( maxU < U[i] ) maxU = U[i];
			if ( minV > V[i] ) minV = V[i];
			if ( maxV < V[i] ) maxV = V[i];
		}

		// 可視化
		double rangeUinv = ( L - 1 ) / ( maxU - minU );
		double rangeVinv = ( L - 1 ) / ( maxV - minV );
		BufferedImage imageU = imageU_;
		BufferedImage imageV = imageV_;
		for ( int i = 0, y = 0; y < N; ++y )
		{
			for ( int x = 0; x < N; ++x, ++i )
			{
				imageU.setRGB( x, y, color[ (int)( ( U[i] - minU ) * rangeUinv ) ] );
				imageV.setRGB( x, y, color[ (int)( ( V[i] - minV ) * rangeVinv ) ] );
			}
		}

		g.drawImage( imageU, 0, 0, null );
		g.drawImage( imageV, N, 0, null );
	}
}
