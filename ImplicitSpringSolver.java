// 常微分方程式の数値解法
//   陽的解法
//   陰的解法
//   クランク・ニコルソン法（陰と陽の折衷手法）
// <applet code="ImplicitSpringSolver" width="640" height="480"></applet>
public final class ImplicitSpringSolver extends java.applet.Applet
		implements Runnable, java.awt.event.MouseListener {
	private java.awt.Image backbuffer_;
	private java.awt.Graphics2D graphics_context_;
	private volatile Thread kicker_;

	private static final float kMass = 10.0f;
	private static final float kSpringConstant = 10000.0f;
	private static final float kDamperConstant = 10.0f;

	private float x_; // 位置
	private float v_; // 速度

	private volatile boolean hasConstrain_;
	private float c_; // 制約位置

	private enum SolverType {
		kExplicit,
		kImplicit,
		kCrankNicolson,
	}

	SolverType type_ = SolverType.kExplicit;

	@Override
	public void init() {
		backbuffer_ = this.createImage(getWidth(), getHeight());
		graphics_context_ = (java.awt.Graphics2D)backbuffer_.getGraphics();
		addMouseListener(this);
	}

	@Override
	public void start() {
		if (kicker_ == null) {
			kicker_ = new Thread(this);
			kicker_.start();
		}
	}

	@Override
	public void stop() {
		kicker_ = null;
	}

	@Override
	public void destroy() {
		removeMouseListener(this);
		if (graphics_context_ != null) {
			graphics_context_.dispose();
			graphics_context_ = null;
		}
		backbuffer_ = null;
	}

	@Override
	public void update(java.awt.Graphics g) {
		paint(g);
	}

	@Override
	public void paint(java.awt.Graphics g) {
		g.drawImage(backbuffer_, 0, 0, null);
	}

	public void run() {
		final Thread thisThread = Thread.currentThread();
		try {
			while (thisThread == kicker_) {
				if (hasConstrain_) {
					x_ = c_;
					v_ = 0.0f;
				} else {
					move(50.0f/1000.0f);
				}
				draw(graphics_context_);
				this.repaint();
				Thread.sleep(33);
			}
		} catch (InterruptedException e) {
			kicker_ = null;
		}
	}

	private void move(final float dt) {
		switch (type_) {
		case kExplicit:
			{
				final float f = (-kSpringConstant * x_ - kDamperConstant * v_) / kMass;
				final float v = v_ + f * dt;
				final float x = x_ + v * dt;

				v_ = v;
				x_ = x;
			}
			break;
		case kImplicit:
			{
				final float a = kDamperConstant / kMass * dt + 1;
				final float b = kSpringConstant / kMass * dt;
				final float c = -dt;
				final float d = 1;

				final float det = a * d - b * c;

				final float e = v_;
				final float f = x_;

				final float v = (d * e - b * f) / det;
				final float x = (a * f - c * e) / det;

				x_ = x;
				v_ = v;
			}
			break;
		case kCrankNicolson:
			{
				final float a = kDamperConstant / kMass * dt * 0.5f + 1.0f;
				final float b = kSpringConstant / kMass * dt * 0.5f;
				final float c = -0.5f * dt;
				final float d = 1.0f;

				final float det = a * d - b * c;

				final float m00 = 1.0f - kDamperConstant / kMass * dt * 0.5f;
				final float m01 =      - kSpringConstant / kMass * dt * 0.5f;
				final float m10 =                                  dt * 0.5f;
				final float m11 = 1.0f;

				final float e = m00 * v_ + m01 * x_;
				final float f = m10 * v_ + m11 * x_;

				final float v = (d * e - b * f) / det;
				final float x = (a * f - c * e) / det;

				x_ = x;
				v_ = v;
			}
			break;
		}
	}

	private void draw(java.awt.Graphics2D g) {
		g.clearRect(0, 0, getWidth(), getHeight());
		g.fillRect(getWidth()/2-50/2+(int)x_, getHeight()/2-50/2, 50, 50);

		switch (type_) {
		case kExplicit: g.drawString("kExplicit", 0, 10); break;
		case kImplicit: g.drawString("kImplicit", 0, 10); break;
		case kCrankNicolson: g.drawString("kCrankNicolson", 0, 10); break;
		}
	}

	public void mouseClicked(java.awt.event.MouseEvent e) {}

	public void mouseEntered(java.awt.event.MouseEvent e) {
		switch (type_) {
		case kExplicit: type_ = SolverType.kImplicit; break;
		case kImplicit: type_ = SolverType.kCrankNicolson; break;
		case kCrankNicolson: type_ = SolverType.kExplicit; break;
		}
	}

	public void mouseExited(java.awt.event.MouseEvent e) {}

	public void mousePressed(java.awt.event.MouseEvent e) {
		hasConstrain_ = true;
		c_ = e.getX() - getWidth() / 2;
	}

	public void mouseReleased(java.awt.event.MouseEvent e) {
		hasConstrain_ = false;
		c_ = e.getX() - getWidth() / 2;
	}
}
