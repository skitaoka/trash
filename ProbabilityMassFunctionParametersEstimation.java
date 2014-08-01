// 離散分布のパラメータを勾配法で計算する
// 確率変数 x は3値をとる: x={-1,0,1}
// 確率質量関数 f(x;a,b,c) = p^{x(x-1)/2} q^{(1-x)(1+x)} r^{x(x+1)/2} を定義する
//  p: x=-1 となる確率 (推定したいパラメータ) ... p = e^a / (e^a + e^b + e^c) と定義する
//  q: x= 0 となる確率 (推定したいパラメータ) ... q = e^b / (e^a + e^b + e^c) と定義する
//  r: x= 1 となる確率 (推定したいパラメータ) ... r = e^c / (e^a + e^b + e^c) と定義する
// 対数尤度 log[f(x;a,b,c)] を最大にするパラメータ a,b,c を求めることで p,q,r を推定する
//  勾配: d log[f(x;a,b,c)] / da = 0.5 x (x - 1) - p
//        d log[f(x;a,b,c)] / db =     (1 - x^2) - q
//        d log[f(x;a,b,c)] / dc = 0.5 x (x + 1) - r
class ProbabilityMassFunctionParametersEstimation {
  public static void main(String[] args) {
    // 推定したい内部パラメータ {-1 がでる確率, 0 がでる確率, 1 がでる確率}
    double[] p = new double [] {0.2, 0.1, 0.7};

    // 累積密度関数を構築
    double[] c = new double[p.length + 1];
    for (int i = 0, length = p.length; i < length; ++i) {
      c[i+1] = c[i] + p[i];
    }

    // 観測データをサンプリング
    int sample_size = 10000;
    int[] x = new int[sample_size];
    for (int i = 0; i < sample_size; ++i) {
      double xi = Math.random();
      int index = 0;
      while (xi > c[++index]);
      x[i] = index - 2;
    }

    // 頻度分布を計算
    System.out.println("頻度分布で計算");
    double[] f = new double[p.length];
    for (int i = 0; i < sample_size; ++i) {
      f[x[i]+1] ++;
    }
    for (int i = 0, length = f.length; i < length; ++i) {
      f[i] /= sample_size;
    }
    for (int i = 0, length = f.length; i < length; ++i) {
      System.out.println(f[i]);
    }

    // 確率的勾配法で計算
    System.out.println("確率的勾配法で計算");
    double[] y = new double[p.length]; // 推定値
    y[0] = Math.log(p[0]);
    y[1] = Math.log(p[1]);
    y[2] = Math.log(p[2]);
    for (int epoch = 1; epoch <= 1000; ++epoch) {
      double alpha = 1.0 / (epoch + 1) / sample_size; // 学習率

      for (int i = 0; i < sample_size; ++i) {
        double ea = Math.exp(y[0]);
        double eb = Math.exp(y[1]);
        double ec = Math.exp(y[2]);
        double ed = 1.0 / (ea + eb + ec);

        // 勾配を計算
        double da = 0.5 * x[i] * (x[i] - 1) - ea * ed;
        double db =       (1 - x[i] * x[i]) - eb * ed;
        double dc = 0.5 * x[i] * (x[i] + 1) - ec * ed;

        // ちょっと更新
        y[0] += alpha * da;
        y[1] += alpha * db;
        y[2] += alpha * dc;
      }
    }
    {
      for (int i = 0, length = f.length; i < length; ++i) {
        y[i] = Math.exp(y[i]);
      }
      double sum = 0;
      for (int i = 0, length = f.length; i < length; ++i) {
        sum += y[i];
      }
      for (int i = 0, length = f.length; i < length; ++i) {
        y[i] /= sum;
      }
      for (int i = 0, length = f.length; i < length; ++i) {
        System.out.println(y[i]);
      }
    }

    // 勾配法 (ラグランジュの未定乗数法) で計算
    System.out.println("ラグランジュの未定乗数法で計算");
    double[] z = new double[p.length]; // 推定値
    z[0] = Math.log(p[0]);
    z[1] = Math.log(p[1]);
    z[2] = Math.log(p[2]);
    double lambda = -0.1;
    for (int epoch = 1; epoch <= 1000; ++epoch) {
      double alpha = 1.0 / (epoch + 1) / sample_size; // 学習率

      double da = 0.0;
      double db = 0.0;
      double dc = 0.0;
      double dl = 0.0;

      for (int i = 0; i < sample_size; ++i) {
        double ea = Math.exp(z[0]);
        double eb = Math.exp(z[1]);
        double ec = Math.exp(z[2]);

        // 勾配を計算
        da += 0.5 * x[i] * (x[i] - 1) + lambda * ea;
        db +=       (1 - x[i] * x[i]) + lambda * eb;
        dc += 0.5 * x[i] * (x[i] + 1) + lambda * ec;
        dl += (ea + eb + ec) - 1;
      }

      // ちょっと更新
      z[0]   += alpha * da;
      z[1]   += alpha * db;
      z[2]   += alpha * dc;
      lambda -= alpha * dl;
    }
    {
      for (int i = 0, length = f.length; i < length; ++i) {
        z[i] = Math.exp(z[i]);
      }
      double sum = 0;
      for (int i = 0, length = f.length; i < length; ++i) {
        sum += z[i];
      }
      for (int i = 0, length = f.length; i < length; ++i) {
        z[i] /= sum;
      }
      for (int i = 0, length = f.length; i < length; ++i) {
        System.out.println(z[i]);
      }
    }
  }
}

