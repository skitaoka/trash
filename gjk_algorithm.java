import java.awt.*;
import java.awt.event.*;
import java.util.*;

// <applet code="gjk_algorithm" width="320" height="240"></applet>
public final class gjk_algorithm extends java.applet.Applet implements Runnable, KeyListener, MouseMotionListener
{
	private volatile Thread kicker_;
	private int width_;
	private int height_;
	private Image backbuffer_;
	private Graphics offscreen_;
	private int fps_;

	// 球データ
	private static final double radius = 50;

	// ポリゴンデータ
	private static final vector2[] vert =
		{
			new vector2( -50, -25 ),
			new vector2(  50, -25 ),
			new vector2(  50,  25 ),
			new vector2( -50,  25 ),
/*
			new vector2( -50, -25 ),
			new vector2( -50,  25 ),
			new vector2(   0,  50 ),
			new vector2(  50,  25 ),
			new vector2(  50, -25 ),
			new vector2(   0, -50 ),
*/
		};

	// mouse control
	private convex a_;
	private affine_transform a2w_;
	private double theta_;

	// animate
	private convex b_;
	private affine_transform b2w_;
	private double phi_;
	private boolean is_rotation_;
	private boolean is_sphere_ = true;

	// レンダリング用データ
	private final vector2[] tempVert_;

	// GJK用
	private final double[][] dp_ = new double[3][3];
	private final double[][] det_ = new double[8][3];
	private final vector2[] p_ = { new vector2(), new vector2(), new vector2() };
	private final vector2[] q_ = { new vector2(), new vector2(), new vector2() };
	private final vector2[] y_ = { new vector2(), new vector2(), new vector2() };
	private int bits_;
	private int all_bits_;
	private int last_bit_;
	private int last_;

	// GJKの結果
	private final vector2 v_ = new vector2();
	private final vector2 pa_ = new vector2();
	private final vector2 pb_ = new vector2();

	// EPA用
	private final PriorityQueue<simplex> queue_ = new PriorityQueue<simplex>();

	// 計算状況
	private boolean is_intersection_;
	private int gjk_iteration_count_;
	private int epa_iteration_count_;

	// 計算用のテンポラリ
	private final vector2 nv = new vector2();
	private final vector2 av = new vector2();
	private final vector2 bv = new vector2();
	private final vector2 w = new vector2();

	public gjk_algorithm()
	{
		a_ = new sphere( radius );
		a2w_ = new affine_transform();
		b_ = new polytope( vert );
		b2w_ = new affine_transform();
		b2w_.translate( 160, 120 );
		tempVert_ = new vector2[ vert.length ];
		for ( int i = 0; i < tempVert_.length; ++i ) {
			tempVert_[i] = new vector2();
		}
	}

	public void init()
	{
		Dimension d = getSize();
		backbuffer_ = createImage( width_ = d.width, height_ = d.height );
		offscreen_ = backbuffer_.getGraphics();

		addKeyListener( this );
		addMouseMotionListener( this );
	}

	public void start()
	{
		if ( kicker_ == null ) {
			kicker_ = new Thread( this );
			kicker_.setDaemon( true );
			kicker_.start();
		}
	}

	public void stop()
	{
		kicker_ = null;
	}

	public void destroy()
	{
		removeKeyListener( this );
		removeMouseMotionListener( this );
	}

	public void update( Graphics g )
	{
		paint( g );
	}

	public void paint( Graphics g )
	{
		g.drawImage( backbuffer_, 0, 0, null );
	}

	public void run()
	{
		try {
			Thread thisThread = Thread.currentThread();
			long time = System.currentTimeMillis();
			int frames = 0;
			while ( kicker_ == thisThread ) {
				long t = System.currentTimeMillis();
				long dt = t - time;
				++frames;
				if ( dt >= 1000 ) {
					fps_ = (int)( 1000 * frames / dt );
					time = t;
					frames = 0;
				}

				advanceFrame();
				render( offscreen_ );
				repaint();
				Thread.sleep( 10 );
			}
		} catch ( InterruptedException e ) {
		} finally {
			kicker_ = null;
		}
	}

	private void advanceFrame()
	{
		if ( is_rotation_ ) {
			theta_ += 0.0001;
			a2w_.rotation( theta_ );

			phi_ += 0.0002;
			b2w_.rotation( phi_ );
		}

		is_intersection_ = true;
		gjk_iteration_count_ = 0;
		epa_iteration_count_ = 0;

		closest_points( a_, a2w_, b_, b2w_, v_, pa_, pb_ );
		if ( is_intersection_ ) {
			expanding_polytope_algorithm( a_, a2w_, b_, b2w_, pa_, pb_ );
		}
	}

	private void render( Graphics g )
	{
		g.setColor( Color.white );
		g.fillRect( 0, 0, width_, height_ );

		a2w_.transform( tempVert_[0], pa_ );
		b2w_.transform( tempVert_[1], pb_ );
		if ( is_intersection_ ) {
			g.setColor( Color.magenta );
		} else {
			g.setColor( Color.green );
		}
		g.drawLine(
				(int)tempVert_[0].x, (int)tempVert_[0].y,
				(int)tempVert_[1].x, (int)tempVert_[1].y );
		g.setColor( Color.black );
		g.fillOval( (int)( tempVert_[0].x - 4 ), (int)( tempVert_[0].y - 4 ), 8, 8 );
		g.setColor( Color.gray );
		g.fillOval( (int)( tempVert_[1].x - 4 ), (int)( tempVert_[1].y - 4 ), 8, 8 );

		g.setColor( Color.red );
		if ( is_sphere_ ) {
			g.drawOval(
					(int)( a2w_.ref_c().x - radius ),
					(int)( a2w_.ref_c().y - radius ),
					(int)( radius * 2 ), (int)( radius * 2 ) );
		} else {
			renderVert( g, a2w_ );
		}
		g.setColor( Color.blue );
		renderVert( g, b2w_ );

		g.setColor( Color.black );
		g.drawString( "fps : " + fps_, 0, 10 );
		g.drawString( "intersection : " + is_intersection_, 0, 20 );
		g.drawString( "gjk iteration : " + gjk_iteration_count_, 0, 30 );
		g.drawString( "epa iteration : " + epa_iteration_count_, 0, 40 );
	}

	private void renderVert( Graphics g, affine_transform o2w )
	{
		for ( int i = 0; i < vert.length; ++i ) {
			o2w.transform( tempVert_[i], vert[i] );
		}
		{
			int i;
			for ( i = 1; i < vert.length; ++i ) {
				g.drawLine(
						(int)tempVert_[i-1].x, (int)tempVert_[i-1].y,
						(int)tempVert_[i].x, (int)tempVert_[i].y );
			}
			{
				g.drawLine(
						(int)tempVert_[i-1].x, (int)tempVert_[i-1].y,
						(int)tempVert_[0].x, (int)tempVert_[0].y );
			}
		}
	}

	public void mouseMoved( MouseEvent e )
	{
	}

	public void mouseDragged( MouseEvent e )
	{
		a2w_.translate( e.getX(), e.getY() );
	}

	public void keyPressed( KeyEvent e )
	{
		switch ( e.getKeyCode() ) {
		case KeyEvent.VK_I:
			a2w_.init();
			b2w_.init();
			break;
		case KeyEvent.VK_R:
			is_rotation_ = !is_rotation_;
			break;
		case KeyEvent.VK_M:
			if ( is_sphere_ ) {
				is_sphere_ = false;
				a_ = new polytope( vert );
			} else {
				is_sphere_ = true;
				a_ = new sphere( radius );
			}
		}
	}

	public void keyReleased( KeyEvent e )
	{
	}

	public void keyTyped( KeyEvent e )
	{
	}

	private void compute_det()
	{
		int i = last_, si = last_bit_;

		for ( int j = 0, sj = 1; j < 3; ++j, sj <<= 1 ) {
			if ( ( sj & bits_ ) != 0 ) {
				dp_[i][j] = dp_[j][i] = y_[i].dot( y_[j] );
			}
		}
		dp_[i][i] = y_[i].dot( y_[i] );

		det_[si][i] = 1;
		for ( int j = 0, sj = 1; j < 3; ++j, sj <<= 1 ) {
			if ( ( sj & bits_ ) != 0 ) {
				det_[si|sj][j] = dp_[i][i] - dp_[i][j];
				det_[si|sj][i] = dp_[j][j] - dp_[j][i];
			}
		}
		if ( all_bits_ == 7 ) {
			det_[7][0] = det_[6][1] * ( dp_[1][1] - dp_[1][0] ) + det_[6][2] * ( dp_[2][1] - dp_[2][0] );
			det_[7][1] = det_[5][0] * ( dp_[0][0] - dp_[0][1] ) + det_[5][2] * ( dp_[2][0] - dp_[2][1] );
			det_[7][2] = det_[3][0] * ( dp_[0][0] - dp_[0][2] ) + det_[3][1] * ( dp_[1][0] - dp_[1][2] );
		}
	}

	private void compute_vector( vector2 v, int s )
	{
		v.init( 0, 0 );
		double sum = 0;
		for ( int i = 0, si = 1; i < 3; ++i, si <<= 1 ) {
			if ( ( si & s ) != 0 ) {
				sum += det_[s][i];
				v.x += y_[i].x * det_[s][i];
				v.y += y_[i].y * det_[s][i];
			}
		}
		sum = 1 / sum;
		v.x *= sum;
		v.y *= sum;
	}

	private void compute_points( vector2 p, vector2 q, int s )
	{
		p.init( 0, 0 );
		q.init( 0, 0 );
		double sum = 0;
		for ( int i = 0, si = 1; i < 3; ++i, si <<= 1 ) {
			if ( ( si & s ) != 0 ) {
				sum += det_[s][i];
				p.x += p_[i].x * det_[s][i];
				p.y += p_[i].y * det_[s][i];
				q.x += q_[i].x * det_[s][i];
				q.y += q_[i].y * det_[s][i];
			}
		}
		sum = 1 / sum;
		p.x *= sum;
		p.y *= sum;
		q.x *= sum;
		q.y *= sum;
	}

	//---------------------------------------------------------------------------
	// GJK
	//---------------------------------------------------------------------------
	private boolean valid( int s )
	{
		for ( int i = 0, si = 1; i < 3; ++i, si <<= 1 ) {
			if ( ( si & all_bits_ ) != 0 ) {
				if ( ( si & s ) != 0 ) {
					if ( det_[s][i] <= 0 ) {
						return false;
					}
				} else {
					if ( det_[si|s][i] > 0 ) {
						return false;
					}
				}
			}
		}
		return true;
	}

	private boolean closest( vector2 v )
	{
		compute_det();
		for ( int s = bits_; s != 0; --s ) {
			if ( ( s & bits_ ) == s ) {
				if ( valid( s | last_bit_ ) ) {
					bits_ = s | last_bit_;
					compute_vector( v, bits_ );
					return true;
				}
			}
		}
		if ( valid( last_bit_ ) ) {
			bits_ = last_bit_;
			v.init( y_[last_] );
			return true;
		}

		throw new RuntimeException();

		//return false; // assertしこむほうがいいよね
	}

	private boolean degenerate( vector2 w )
	{
		for ( int i = 0, si = 1; i < 4; ++i, si <<= 1 ) {
			if ( ( ( all_bits_ & si ) != 0 ) && ( w.equals( y_[i] ) ) ) {
				return true;
			}
		}
		return false;
	}

	private void closest_points(
			convex a, affine_transform a2w,
			convex b, affine_transform b2w,
			vector2 v, vector2 pa, vector2 pb )
	{
		{// 適当にvを初期化
			vector2 x = a2w.ref_c();
			vector2 y = b2w.ref_c();
			v.init( x.x - y.x, x.y - y.y );
		}

		double dist = v.norm2();
		double mu = 0;
		double dot;

		bits_ = 0;
		all_bits_ = 0;

		while ( ( bits_ < 7 ) && ( dist > 1e-10 ) ) {
			++gjk_iteration_count_;

			for ( last_ = 0, last_bit_ = 1; ( bits_ & last_bit_ ) != 0; ++last_, last_bit_ <<= 1 ) {}

			nv.init( -v.x, -v.y );
			a2w.transpose( av, nv ); a.support( p_[last_], av ); a2w.transform( av, p_[last_] );
			b2w.transpose( bv, v ); b.support( q_[last_], bv ); b2w.transform( bv, q_[last_] );
			w.init( av.x - bv.x, av.y - bv.y );

			dot = v.dot( w );
			if ( dot > 0 ) {
				is_intersection_ = false;
			}

			mu = Math.max( mu, dot / dist );
			if ( dist - mu <= dist * 1e-6 ) {
				break;
			}

			if ( degenerate( w ) ) {
				break;
			}

			y_[last_].init( w );
			all_bits_ = bits_ | last_bit_;

			if ( !closest( v ) ) {
				break;
			}

			dist = v.norm2();
		}

		compute_points( pa, pb, bits_ );
	}

	//---------------------------------------------------------------------------
	// EPA
	//---------------------------------------------------------------------------
	private boolean is_empty_queue()
	{
		return queue_.isEmpty();
	}

	private void enqueue( int _1, int _2 )
	{
		int s = ( 1 << _1 ) | ( 1 << _2 );
		for ( int i = 0, si = 1; i < 3; ++i, si <<= 1 ) {
			if ( ( si & all_bits_ ) != 0 ) {
				if ( ( si & s ) != 0 ) {
					if ( det_[s][i] < 0 ) {
						return;
					}
				}
			}
		}
		simplex obj = new simplex( y_[_1], y_[_2], p_[_1], p_[_2], q_[_1], q_[_2] );
		compute_vector( obj.v, s );
		obj.dist = obj.v.norm2();
		compute_points( obj.pa, obj.pb, s );
		queue_.add( obj );
	}

	private simplex dequeue()
	{
		return queue_.poll();
	}

	private void simplex_make()
	{
		bits_ = 0;
		all_bits_ = 0;
		for ( last_ = 0, last_bit_ = 1; last_ < 3; ++last_, last_bit_ <<= 1 )
		{
			all_bits_ |= last_bit_;
			compute_det();
			bits_ |= last_bit_;
		}

		enqueue( 0, 1 );
		enqueue( 0, 2 );
	}

	private void expanding_polytope_algorithm(
			convex a, affine_transform a2w,
			convex b, affine_transform b2w,
			vector2 pa, vector2 pb )
	{
		if ( bits_ != 7 )
		{
			vector2 y1 = null, y2 = null;
			vector2 p1 = null, p2 = null;
			vector2 q1 = null, q2 = null;
			switch ( bits_ )
			{
			case 1: pa.init( p_[0] ); pb.init( q_[0] ); return;
			case 2: pa.init( p_[1] ); pb.init( q_[1] ); return;
			case 4: pa.init( p_[2] ); pb.init( q_[2] ); return;
			case 3: y1 = y_[0]; y2 = y_[1]; p1 = p_[0]; p2 = p_[1]; q1 = q_[0]; q2 = q_[1]; break;
			case 5: y1 = y_[0]; y2 = y_[2]; p1 = p_[0]; p2 = p_[2]; q1 = q_[0]; q2 = q_[2]; break;
			case 6: y1 = y_[1]; y2 = y_[2]; p1 = p_[1]; p2 = p_[2]; q1 = q_[1]; q2 = q_[2]; break;
			default: System.out.println( "中心がそろって方向が出ません" ); return;
			}

			if ( y1.norm2sq() < y2.norm2sq() ) {
				pa.init( p1 );
				pb.init( q1 );
			} else {
				pa.init( p2 );
				pb.init( q2 );
			}
			return;
		}

		queue_.clear();
		enqueue( 0, 1 );
		enqueue( 0, 2 );
		enqueue( 1, 2 );

		for ( int i = 0; !is_empty_queue(); ++i )
		{
			++epa_iteration_count_;

			simplex s = dequeue();

			a2w.transpose( av, s.v ); a.support( p_[0], av ); a2w.transform( av, p_[0] );
			nv.init( -s.v.x, -s.v.y );
			b2w.transpose( bv, nv ); b.support( q_[0], bv ); b2w.transform( bv, q_[0] );
			w.init( av.x - bv.x, av.y - bv.y );

			double mu = s.v.dot( w ) / s.dist;
			if ( mu - s.dist < mu * 1e-12 )
			{
				pa.init( s.pa );
				pb.init( s.pb );
				return;
			}

			y_[0].init( w );
			y_[1].init( s.y1 );
			y_[2].init( s.y2 );
			p_[1].init( s.p1 );
			p_[2].init( s.p2 );
			q_[1].init( s.q1 );
			q_[2].init( s.q2 );

			simplex_make();
		}
	}

	private static final class simplex implements Comparable<simplex>
	{
		vector2 y1, y2;
		vector2 p1, p2;
		vector2 q1, q2;
		vector2 v;
		double dist;
		vector2 pa, pb;

		simplex( vector2 y1, vector2 y2, vector2 p1, vector2 p2, vector2 q1, vector2 q2 )
		{
			this.y1 = new vector2( y1 );
			this.y2 = new vector2( y2 );
			this.p1 = new vector2( p1 );
			this.p2 = new vector2( p2 );
			this.q1 = new vector2( q1 );
			this.q2 = new vector2( q2 );
			this.v = new vector2();
			this.pa = new vector2();
			this.pb = new vector2();
		}

		public int compareTo( simplex s )
		{
			if ( dist < s.dist ) {
				return -1;
			}
			if ( dist > s.dist ) {
				return 1;
			}
			return 0;
		}
	}

	private static abstract class convex
	{
		abstract void support( vector2 w, vector2 v );
	}

	private static final class sphere extends convex
	{
		private double radius_;

		sphere( double radius )
		{
			radius_ = radius;
		}

		void support( vector2 w, vector2 v )
		{
			double norm2sq = v.norm2sq();
			if ( norm2sq > 1e-6 ) {
				double s = radius_ / Math.sqrt( norm2sq );
				w.init( v.x * s, v.y * s );
			} else {
				w.init( 0, 0 );
			}
		}
	}

	private static final class polytope extends convex
	{
		private final vector2[] vert_;

		polytope( vector2[] vert )
		{
			vert_ = vert;
		}

		void support( vector2 w, vector2 v )
		{
			int idx = 0;
			double max = -Double.MAX_VALUE;
			for ( int i = 0; i < vert_.length; ++i ) {
				double dot = v.dot( vert_[i] );
				if ( max < dot ) {
					max = dot;
					idx = i;
				}
			}
			w.init( vert_[idx] );
		}
	}

	private static final class affine_transform
	{
		private final matrix2 b_; // 姿勢
		private final vector2 c_; // 平行移動

		affine_transform()
		{
			b_ = new matrix2();
			b_.identity();
			c_ = new vector2();
		}

		void init()
		{
			b_.identity();
		}

		// 回転
		void rotation( double theta )
		{
			b_.rotation( theta );
		}

		// 平行移動
		void translate( double x, double y )
		{
			c_.x = x;
			c_.y = y;
		}

		vector2 ref_c()
		{
			return c_;
		}

		// y = x * B + c
		void transform( vector2 y, vector2 x )
		{
			y.x = x.x * b_._11 + x.y * b_._21 + c_.x;
			y.y = x.x * b_._12 + x.y * b_._22 + c_.y;
		}

		// y = x * B^T
		void transpose( vector2 y, vector2 x )
		{
			y.x = x.x * b_._11 + x.y * b_._12;
			y.y = x.x * b_._21 + x.y * b_._22;
		}
	}

	private static final class vector2
	{
		double x;
		double y;

		vector2()
		{
		}

		vector2( double x, double y )
		{
			this.x = x;
			this.y = y;
		}

		vector2( vector2 v )
		{
			x = v.x;
			y = v.y;
		}

		void init( vector2 v )
		{
			x = v.x;
			y = v.y;
		}

		void init( double x, double y )
		{
			this.x = x;
			this.y = y;
		}

		double dot( vector2 v )
		{
			return x * v.x + y * v.y;
		}

		double norm2sq()
		{
			return dot( this );
		}

		double norm2()
		{
			return Math.sqrt( norm2sq() );
		}

		@Override
		public boolean equals( Object obj )
		{
			return ( obj instanceof vector2 ) && ( equals( (vector2)obj ) );
		}

		public boolean equals( vector2 v )
		{
			return ( x == v.x ) && ( y == v.y );
		}
	}

	private static final class matrix2
	{
		double _11, _12;
		double _21, _22;

		void identity()
		{
			_11 = 1; _12 = 0;
			_21 = 0; _22 = 1;
		}

		void rotation( double theta )
		{
			double sin = fast_math.sin( theta );
			double cos = fast_math.cos( theta );
			_11 =  cos; _12 = sin;
			_21 = -sin; _22 = cos;
		}
	}

	private static final class fast_math
	{
		private static final double[] sinA;
		private static final double[] sinB;
		private static final double[] cosA;
		private static final double[] cosB;

		static
		{
			sinA = new double[256];
			sinB = new double[256];
			cosA = new double[256];
			cosB = new double[256];
			for ( int i = 0; i < 256; ++i ) {
				double a = 2 * Math.PI * ( 1.0 / ( 1 << 16 ) ) * ( i	  );
				double b = 2 * Math.PI * ( 1.0 / ( 1 << 16 ) ) * ( i << 8 );
				sinA[i] = Math.sin( a );
				sinB[i] = Math.sin( b );
				cosA[i] = Math.cos( a );
				cosB[i] = Math.cos( b );
			}
		}

		static double sin( double x )
		{
			int i = (int)( ( 1.0 / ( 2 * Math.PI ) ) * ( 1 << 16 ) * x ) & ( ( 1 << 16 ) - 1 );
			int a = i >> 8;
			int b = i & ( ( 1 << 8 ) - 1 );
			return sinA[a] * cosB[b] + cosA[a] * sinB[b];
		}

		static double cos( double x )
		{
			int i = (int)( ( 1.0 / ( 2 * Math.PI ) ) * ( 1 << 16 ) * x ) & ( ( 1 << 16 ) - 1 );
			int a = i >> 8;
			int b = i & ( ( 1 << 8 ) - 1 );
			return cosA[a] * cosB[b] - sinA[a] * sinB[b];
		}
	}
}
