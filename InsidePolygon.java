import java.awt.*;
import java.awt.geom.*;
import java.awt.event.*;

// <applet code="InsidePolygon" width="800" height="600"></applet>
public final class InsidePolygon extends java.applet.Applet {
	private static final class Vec2 {
		double x;
		double y;

		double dot(final Vec2 v) {
			return x * v.x + y * v.y;
		}

		double cross(final Vec2 a) {
			return x * a.y - y * a.x;
		}

		Vec2 sub(final Vec2 a, final Vec2 b) {
			x = a.x - b.x;
			y = a.y - b.y;
			return this;
		}

        Vec2 normalize() {
            final double dot = this.dot(this);
            if (dot > 0.0) {
              final double invLen = 1.0 / Math.sqrt(dot);
              x *= invLen;
              y *= invLen;
            }
            return this;
        }
	}

	private final Vec2 v1 = new Vec2();
	private final Vec2 v2 = new Vec2();
	private final Vec2 v3 = new Vec2();

	private final Vec2[] points = new Vec2[100];
	private final Color[] colors = new Color[points.length];

	private final Vec2 a = new Vec2();
	private final Vec2 b = new Vec2();
	private final Vec2 c = new Vec2();
	private double area;

	@Override
	public void init() {
		v1.x = 150;
		v1.y = 150;
		v2.x = 500;
		v2.y = 300;
		v3.x = 180;
		v3.y = 500;

		area = 0.5 * (v1.cross(v2) + v2.cross(v3) + v3.cross(v1));

        for (int i = 0, length = points.length; i < length; ++i) {
          points[i] = new Vec2();
          colors[i] = Color.black;
        }

		addMouseMotionListener(new MouseMotionAdapter() {
			@Override
			public void mouseDragged(final MouseEvent evt) {
                for (final Vec2 v : points) {
                  v.x = Math.random() * getWidth ();
                  v.y = Math.random() * getHeight();
                }
				points[0].x = evt.getX();
				points[0].y = evt.getY();

                for (int i = 0, length = points.length; i < length; ++i) {
                  a.sub(v1, points[i]).normalize();
                  b.sub(v2, points[i]).normalize();
                  c.sub(v3, points[i]).normalize();
                  final double value
                  	  = Math.acos(a.dot(b))
                      + Math.acos(b.dot(c))
                      + Math.acos(c.dot(a)) - 2.0 * Math.PI;
                  colors[i] = (Math.abs(value) < 1e-9) ? Color.red : Color.blue;
                }

				repaint();
			}
		});
	}

	@Override
	public void update(Graphics g) {
	  paint(g);
	}

	@Override
	public void paint(Graphics g) {
        g.setColor(Color.black);

		g.drawString("area: " + area, 0, 20);

		g.drawString("v1", (int) v1.x, (int) v1.y);
		g.drawOval((int) v1.x, (int) v1.y, 10, 10);
		g.drawString("v2", (int) v2.x, (int) v2.y);
		g.drawOval((int) v2.x, (int) v2.y, 10, 10);
		g.drawString("v3", (int) v3.x, (int) v3.y);
		g.drawOval((int) v3.x, (int) v3.y, 10, 10);

        g.drawLine((int) v1.x, (int) v1.y, (int) v2.x, (int) v2.y);
        g.drawLine((int) v2.x, (int) v2.y, (int) v3.x, (int) v3.y);
        g.drawLine((int) v3.x, (int) v3.y, (int) v1.x, (int) v1.y);

        for (int i = 0, length = points.length; i < length; ++i) {
          g.setColor(colors[i]);
		  g.fillOval((int) points[i].x, (int) points[i].y, 10, 10);
		}
	}
}
