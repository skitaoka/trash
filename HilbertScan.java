import java.awt.Color;
import java.awt.Graphics;
import java.awt.Image;
import java.awt.Point;

/**
 * <APPLET code="HilbertScan" width="512" height="512">
 * </APPLET>
 */
public final class HilbertScan extends java.applet.Applet implements Runnable {

  /**
   * 画面の大きさ
   */
  private static final int SCREEN_SIZE = 512;

  /**
   * スキャンパターン
   * fr ^ 0x1  // 回転
   * fr ^ 0x2  // フリップ
   * fr ^ 0x3  // 回転＋フリップ
   */
  private static final Point[][] scanPattern =
  {
    { new Point( 0, 0 ), new Point( 1, 0 ), new Point( 1, 1 ), new Point( 0, 1 ) },    // コ
    { new Point( 1, 1 ), new Point( 0, 1 ), new Point( 0, 0 ), new Point( 1, 0 ) },    // ⊂
    { new Point( 0, 0 ), new Point( 0, 1 ), new Point( 1, 1 ), new Point( 1, 0 ) },    // Ц
    { new Point( 1, 1 ), new Point( 1, 0 ), new Point( 0, 0 ), new Point( 0, 1 ) },    // П
  };

  private Color[][] fragmentColor;


  private int _n;      // 画像の大きさの対数 log2( __size )
  private int _imageSize;  // 画像の大きさ（幅と高さが同じ）
  private int _pixelSize;  // ピクセルの大きさ

  private Thread    _kicker;
  private Image    _backBuffer;
  private Graphics  _offscreen;

  public void init() {
    _backBuffer = createImage( SCREEN_SIZE, SCREEN_SIZE );
    _offscreen = _backBuffer.getGraphics();
  }

  public void start() {
    if ( _kicker == null ) {
      _kicker = new Thread( this );
      _kicker.setDaemon( true );
      _kicker.start();
    }
  }

  public void stop() {
    _kicker = null;
  }

  private void destory() {
    _offscreen.dispose();
  }

  public void update( Graphics g ) {
    paint( g );
  }

  public void paint( Graphics g ) {
    g.drawImage( _backBuffer, 0, 0, SCREEN_SIZE, SCREEN_SIZE, null );
  }

  public void run() {
    clear( _offscreen );
    hilbertScan();
    restart();
  }

  private void clear( Graphics g ) {
    g.setColor( Color.white );
    g.fillRect( 0, 0, SCREEN_SIZE, SCREEN_SIZE );

    _n++;
    if ( _n > 6 ) {
      _n = 1;
    }
    _imageSize = 1 << _n;
    _pixelSize = SCREEN_SIZE / _imageSize;

    fragmentColor = new Color[ _imageSize ][ _imageSize ];
    for ( int y = 0; y < _imageSize; y++ ) {
      for ( int x = 0; x < _imageSize; x++ ) {
        fragmentColor[y][x] = new Color(
            x * ( 1.0f / ( _imageSize - 1 ) ),
            ( ( _imageSize - 1 ) - x ) * ( 1.0f / ( _imageSize - 1 ) ),
            y * ( 1.0f / ( _imageSize - 1 ) ) );
      }
    }
  }

  private void hilbertScan() {
    hilbertScan( _n, 0, 0, 0 );
  }

  private void hilbertScan( int i, int fr, int x, int y ) {
    if ( _kicker == null ) {
      return;
    }

    if ( i == 0 ) {
      paint( x, y, _offscreen );
    } else {
      Point[] aScanPattern = scanPattern[fr];
      int j = 1 << ( i - 1 );
      hilbertScan( i - 1, fr ^ 0x2, x + j * aScanPattern[0].x, y + j * aScanPattern[0].y );
      hilbertScan( i - 1, fr      , x + j * aScanPattern[1].x, y + j * aScanPattern[1].y );
      hilbertScan( i - 1, fr      , x + j * aScanPattern[2].x, y + j * aScanPattern[2].y );
      hilbertScan( i - 1, fr ^ 0x3, x + j * aScanPattern[3].x, y + j * aScanPattern[3].y );
    }
  }

  private void paint( int x, int y, Graphics g ) {
    g.setColor( fragmentColor[x][y] );
    g.fillRect( x * _pixelSize, y * _pixelSize, _pixelSize, _pixelSize );

    repaint();

    try {
      Thread.sleep( 400L / _n );
    } catch ( InterruptedException e ) {
    }
  }

  private void restart() {
    try {
      Thread.sleep( 1000L );
    } catch ( InterruptedException e ) {
    }

    stop();

    System.gc();
    System.runFinalization();

    start();
  }
}
