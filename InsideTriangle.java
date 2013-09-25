import java.awt.*;
import java.awt.geom.*;
import java.awt.event.*;

// <applet code="InsideTriangle" width="800" height="600"></applet>
public final class InsideTriangle extends java.applet.Applet {

	private static final class Vec3 {
		double x;
		double y;
		double z;

		double dot(final Vec3 v) {
			return x * v.x + y * v.y + z * v.z;
		}

		Vec3 cross(final Vec3 a, final Vec3 b) {
			x = a.y * b.z - a.z * b.y;
			y = a.z * b.x - a.x * b.z;
			z = a.x * b.y - a.y * b.x;
			return this;
		}

		Vec3 sub(final Vec3 a, final Vec3 b) {
			x = a.x - b.x;
			y = a.y - b.y;
			z = a.z - b.z;
			return this;
		}
	}

	private final Vec3 v1 = new Vec3();
	private final Vec3 v2 = new Vec3();
	private final Vec3 v3 = new Vec3();
	private final Vec3 p  = new Vec3();

	private final Vec3 e = new Vec3();
	private final Vec3 f = new Vec3();

	private final Vec3 x  = new Vec3();
	private final Vec3 y = new Vec3();
	private final Vec3 z = new Vec3();
	private final Vec3 d = new Vec3();

	private double t = 0.0;
	private double u = 0.0;
	private double v = 0.0;

	@Override
	public void init() {
		v1.x = 150;
		v1.y = 150;
		v2.x = 500;
		v2.y = 300;
		v3.x = 180;
		v3.y = 500;

		addMouseListener(new MouseAdapter() {
			@Override
			public void mousePressed(final MouseEvent evt) {
				p.x = evt.getX();
				p.y = evt.getY();

				e.sub(v2, v1);
				f.sub(v3, v1);

				x.cross(f, e);
				y.cross(f, x);
				z.cross(x, e);
				d.sub(v1, p);

				final double det = 1.0 / x.dot(x);
				t = det * x.dot(d);
				u = det * y.dot(d);
				v = det * z.dot(d);

				repaint();
			}
		});
	}

	@Override
	public void paint(Graphics g) {

		g.drawString("v1", (int) v1.x, (int) v1.y);
		g.drawOval((int) v1.x, (int) v1.y, 10, 10);
		g.drawString("v2", (int) v2.x, (int) v2.y);
		g.drawOval((int) v2.x, (int) v2.y, 10, 10);
		g.drawString("v3", (int) v3.x, (int) v3.y);
		g.drawOval((int) v3.x, (int) v3.y, 10, 10);
		g.fillOval((int) p .x, (int) p .y, 10, 10);

		g.drawString("t: " + (int)(t*10000.0)/10000.0, 0, 20);
		g.drawString("u: " + (int)(u*10000.0)/10000.0, 0, 40);
		g.drawString("v: " + (int)(v*10000.0)/10000.0, 0, 60);
	}
}
