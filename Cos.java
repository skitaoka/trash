
/*
cos を近似する式をつくる：
	cos_ap(x) := a * x^4 - b x^2 + 1

J(a,b) = 1/2 (cos_ap(x) - cos(x))^2 を最小化する a と b を求める。

勾配を求める。
	dJ/da =  E x^4
	dJ/db = -E x^2
ここで
	E = cos_ap(x) - cos(x)
	  = a x^4 - b x^2 + 1 - cos(x)
とおいた。

パラメータの更新式を
	a' <- a + t dJ/da
	b' <- b + t dJ/db
とする。

J(a', b') が最小になるような t を直線探索で求める。

J(a', b') = J(t)
          = 1/2 ((a + t dJ/da) x^4 - (b + t dJ/db) x^2 + 1 - cos(x))^2

dJ(t)/dt = F (dJ/da x^4 - dJ/db x^2)

ここで
	F = (a + t dJ/da) x^4 - (b + t dJ/db) x^2 + 1 - cos(x)
とおいた。

dJ(t)/dt = 0 を解くと
	t = -E / (dJ/da x^4 - dJ/db x^2)
が得られる。


これを更新式に代入して、もろもろ整理すると、
	a' <- a - E/((x^4+1)    )
	b' <- b + E/((x^4+1) x^2)
が得られる。
*/
public final class Cos {

  public static void main(final String[] args) {

    double a = 1.0 /  2.0;
    double b = 1.0 / 25.0;

    Cos.test(a, b);

    // 確率勾配法で求める
    final int N = 100000000;
    for (int n = 0; n < N; ++n) {
      final double x = Math.random() * (Math.PI/2); // x をサンプリング
      final double y = Cos.cos(x, a, b) - Math.cos(x);
      final double x2 = x  * x;
      final double x4 = x2 * x2;
      final double x6 = x2 * x4;
      final double da = -y / (x4 + 1.0);
      final double db =  y / (x6 + x2);
      a += da;
      b += db;
    }

    Cos.test(a, b);

    System.out.println(a);
    System.out.println(b);
  }

  // 近似した cos
  private static double cos(final double x, final double a, final double b) {
    final double x2 = x * x;
    return (a * x2 - b) * x2 + 1.0;
  }

  private static void test(final double a, final double b) {
    final int N = 100000;

    double err = 0.0;
    for (int n = 0; n < N; ++n) {
      final double x = Math.random() * (Math.PI/2); // x をサンプリング
      final double c = Math.cos(x);
      final double e = (Cos.cos(x, a, b) - c) / c;
      err += e * e;
    }
    System.out.println(Math.sqrt(err / N));
  }
}
/*
余談
	J(a,b) = 1/2 int_{0}^{pi/2} (cos_ap(x) - cos(x))^2 dx
	       = pi^9/9216 a^2 - pi^7/896 a b + (pi^5/160 - pi^4/16 + 3 pi^2 - 24) a
	       + pi^5/320  b^2                - (pi^3/ 24 - pi^2/ 4 - 3 pi/8 +  2) b
	       - 1
	dJ/da  = pi^9/4608 a - pi^7/896 b + pi^5/160   - pi^4/16 + 3 pi^2   - 24
	dJ/db  =             - pi^7/896 a + pi^5/160 b - pi^3/24 +   pi^2/4 -  2
と定義したほうがよかったかもしれない。

*/