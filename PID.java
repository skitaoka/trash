import java.awt.Image;
import java.awt.Graphics;
import java.awt.geom.Point2D;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionListener;

// <applet code="PID" width="640" height="480"></applet>
public final class PID extends javax.swing.JApplet implements Runnable, MouseMotionListener {

  private static final long sleepTime = 100L;
  private static final double deltaTime = sleepTime / 1000.0;
  private static final double mass = 1.0;
  private static final double e = 0.1;
  private static final double pGain = mass / (10 * deltaTime * deltaTime);
  private static final double vGain
    = Math.sqrt(4.0 * mass * pGain * Math.log(e) * Math.log(e)
            / (Math.PI * Math.PI + Math.log(e) * Math.log(e)));
  private static final double iGain = 1.0;

  private Point2D.Double position = new Point2D.Double();
  private Point2D.Double velocity = new Point2D.Double();
  private Point2D.Double force    = new Point2D.Double();
  private Point2D.Double stress   = new Point2D.Double();
  private double         time     = 0.0;

  private volatile Point2D.Double mousePos = new Point2D.Double();
  private volatile Thread kicker;
  private Image offscreen;
  private Graphics dc;

  @Override
  public void init() {
    offscreen = this.createImage(getWidth(), getHeight());
    dc = offscreen.getGraphics();
    addMouseMotionListener(this);
  }

  @Override
  public void destroy() {
    removeMouseMotionListener(this);
    dc.dispose();
    offscreen.flush();
  }

  @Override
  public void start() {
    if (kicker == null) {
      kicker = new Thread(this);
      kicker.start();
    }
  }

  @Override
  public void stop() {
    kicker = null;
  }

  @Override
  public void run() {
    final Thread thisThread = Thread.currentThread();
    try {
      while (kicker == thisThread) {
        update(deltaTime);
        draw(dc);
        repaint();
        Thread.sleep(sleepTime);
      }
    } catch (final InterruptedException e) {
    } finally {
      kicker = null;
    }
  }

  @Override
  public void paint(final Graphics g) {
    g.drawImage(offscreen, 0, 0, null);
    g.drawRect((int)position.x, (int)position.y, 10, 10);
  }

  private void update(final double dt) {
    final Point2D.Double target = mousePos;
    final double dx = target.x - position.x;
    final double dy = target.y - position.y;

    time     += dt;
    stress.x += dt * dx;
    stress.y += dt * dy;

    force.x = pGain * (dx + stress.x / time) - vGain * velocity.x;
    force.y = pGain * (dy + stress.y / time) - vGain * velocity.y;

    // update velocity
    velocity.x += (force.x / mass) * dt;
    velocity.y += (force.y / mass) * dt;

    // update position
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;
  }

  private void draw(final Graphics g) {
    g.clearRect(0, 0, getWidth(), getHeight());
  }

  @Override
  public void mouseMoved(final MouseEvent evt) {
    mousePos = new Point2D.Double(evt.getX(), evt.getY());
  }

  @Override
  public void mouseDragged(final MouseEvent evt) {
  }
}
