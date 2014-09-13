import java.util.Arrays;
import java.util.Random;

final class Autoencoder {

  private final Random rand = new Random();

  private final Type type;
  private final Unit unit;

  private final double beta; // normalization term

  private final int n; // input dims.
  private final int m; // output dims.

  private final double[][] a; // weight matrix
  private final double[]   b; // offset vector
  private final double[][] c; // weight matrix
  private final double[]   d; // offset vector

  private final double[][] da; // dL/dA
  private final double[]   db; // dL/db
  private final double[][] dc; // dL/dc
  private final double[]   dd; // dL/dd

  Autoencoder(Type type, Unit unit, double beta, int n, int m) {
    this.type = type;
    this.unit = unit;
    this.beta = beta;

    this.n = n;
    this.m = m;

    this.a = new double[m][n];
    this.b = new double[m];
    this.c = new double[n][m];
    this.d = new double[n];

    this.da = new double[m][n];
    this.db = new double[m];
    this.dc = new double[n][m];
    this.dd = new double[n];

    java.util.Random rand = new java.util.Random();
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        a[i][j] = rand.nextGaussian() * 0.1;
      }/* always */{
        b[i]    = rand.nextGaussian() * 0.1;
      }
    }
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < m; ++i) {
        c[j][i] = rand.nextGaussian() * 0.1;
      }/* always */{
        d[j]    = rand.nextGaussian() * 0.1;
      }
    }
  }

  private static double sigmoid(double x) {
    return 1.0 / (1.0 + Math.exp(-x));
  }

  private static enum Type {
    BINOMIAL {
      @Override
      double noise(double x, Random engine) {
        return engine.nextBoolean() ? 1.0 - x : x;
      }

      @Override
      double link(double x) {
        return sigmoid(x);
      }

      // 交差エントロピー: -x log(z) - (1-x) log(1-z)
      @Override
      double error(double x, double z) {
        return
          -        x  * Math.log(      z)
          - (1.0 - x) * Math.log(1.0 - z);
      }
    },

    REAL {
      @Override
      double noise(double x, Random engine) {
        return x + engine.nextGaussian();
      }

      @Override
      double link(double x) {
        return x;
      }

      // 二乗誤差: 1/2 (x - z)^2
      @Override
      double error(double x, double z) {
        return 0.5 * (x - z) * (x - z);
      }
    },

    NONNEGATIVE{
      @Override
      double noise(double x, Random engine) {
        // 平均 x のポアソン分布からサンプリングする
        double a = Math.exp(-x);

        int k = 0;
        for (double xp = engine.nextDouble(); a <= xp; ++k) {
          xp *= engine.nextDouble();
        }
        return k;
      }

      @Override
      double link(double x) {
        return Math.exp(x);
      }

      // I-ダイバージェンス: x log(x/z) + (x - z)
      @Override
      double error(double x, double z) {
        assert(x < 0.0);
        assert(z < 0.0);
/*
        double err = x - z;
        if (x > 0.0) {
          err += x * Math.log(x / z);
        }
        return err;
/*/
        return z - x * Math.log(z);
//*/
      }
    };

    abstract double noise(double x, Random engine);
    abstract double link (double x);
    abstract double error(double x, double z);
  }

  private static enum Unit {
    // sigmoid function
    SIGMOID {
      @Override
      double f(double x) {
        return sigmoid(x);
      }

      @Override
      double g(double x, double y) {
        return y * (1.0 - y);
      }
    },

    // hyperbolic tangent
    TANH {
      @Override
      double f(double x) {
        return Math.tanh(x);
      }

      @Override
      double g(double x, double y) {
        return 1.0 - y * y;
      }
    },

    // ReLU (rectified linear unit)
    RELU {
      @Override
      double f(double x) {
        return (x > 0.0) ? x : 0.0;
      }

      @Override
      double g(double x, double y) {
        return (x > 0.0) ? 1.0 : 0.0;
      }
    },

    // Leaky ReLU
    LREL {
      @Override
      double f(double x) {
        return (x > 0.0) ? x : x * 0.01;
      }

      @Override
      double g(double x, double y) {
        return (x > 0.0) ? 1.0 : 0.01;
      }
    },

    // softplus
    SOFTPLUS {
      @Override
      double f(double x) {
        return Math.log(1.0 + Math.exp(x));
      }

      @Override
      double g(double x, double y) {
        return sigmoid(x);
      }
    };

    // @param x real value
    // @return f(x)
    abstract double f(double x);

    // @param x real value
    // @param y := f(x)
    // @return df(x)/dx
    abstract double g(double x, double y);
  }

  // y = f(A x + b)
  private static double[][] axpb(double[][] a, double[] x, double[] b, Unit unit) {
    int m = a   .length;
    int n = a[0].length;
    double[][] y = new double[2][a.length];
    for (int j = 0; j < m; ++j) {
      double sum = b[j];
      for (int i = 0; i < n; ++i) {
        sum += a[j][i] * x[i];
      }
      y[0][j] = sum;
      y[1][j] = unit.f(sum);
    }
    return y;
  }

  // z = f(C y + d)
  private static double[] cypd(double[][] c, double[] y, double[] d, Type type) {
    int m = c[0].length;
    int n = c   .length;
    double[] z = new double[n];
    for (int i = 0; i < n; ++i) {
      double sum = d[i];
      for (int j = 0; j < m; ++j) {
        sum += c[i][j] * y[j];
      }
      z[i] = type.link(sum);
    }

    return z;
  }

  private static double[] mul(double[][] a, boolean transpose, double[] x) {
    int m = a   .length;
    int n = a[0].length;

    if (transpose) {
      assert(m == x.length);

      double[] y = new double[n];
      for (int j = 0; j < n; ++j) {
        double sum = 0.0;
        for (int i = 0; i < m; ++i) {
          sum += a[i][j] * x[i];
        }
        y[j] = sum;
      }

      return y;
    } else {
      assert(n == x.length);

      double[] y = new double[m];
      for (int i = 0; i < m; ++i) {
        double sum = 0.0;
        for (int j = 0; j < n; ++j) {
          sum += a[i][j] * x[j];
        }
        y[i] = sum;
      }

      return y;
    }
  }

  private static double[][] mul(double[][] a, boolean transpose, double[][] b) {
    int m = a   .length;
    int n = a[0].length;

    if (transpose) {
      assert(m == b.length);

      int l = b[0].length;

      double[][] c = new double[n][l];
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < l; ++j) {
          double sum = 0.0;
          for (int k = 0; k < m; ++k) {
            sum += a[k][i] * b[k][j];
          }
          c[i][j] = sum;
        }
      }

      return c;
    } else {
      assert(n == b.length);

      int l = b[0].length;

      double[][] c = new double[m][l];
      for (int i = 0; i < m; ++i) {
        for (int j = 0; j < l; ++j) {
          double sum = 0.0;
          for (int k = 0; k < n; ++k) {
            sum += a[i][k] * b[k][j];
          }
          c[i][j] = sum;
        }
      }

      return c;
    }
  }

  private static double[] sub(double[] x, double[] z) {
    assert x.length == z.length;
    int n = x.length;
    double[] e = new double[n];
    for (int i = 0; i < n; ++i) {
      e[i] = x[i] - z[i];
    }
    return e;
  }

  // 勾配の値をチェックする
  void test(Random engine) {
    // a
    {
      // ランダムな入力配列を作る
      double[] x = new double[n];
      for (int i = 0; i < n; ++i) {
        x[i] = engine.nextGaussian();
      }

      // 誤差をはかる
      double[][] y = axpb(a, x   , b, unit);
      double[]   z = cypd(c, y[1], d, type);

      // df(A)/dA を作る
      for (int k = 0; k < m; ++k) {
        double sum = 0.0;
        // C^T(x-z) * y * (1-y)
        for (int j = 0; j < n; ++j) {
          sum += c[j][k] * (x[j] - z[j]);
        }
        sum *= unit.g(y[0][k], y[1][k]);

        for (int j = 0; j < n; ++j) {
          da[k][j] = sum * x[j];
        }/* always */{
          db[k]    = sum;
        }
      }

      // df(C)/dC を作る
      for (int j = 0; j < n; ++j) {
        for (int k = 0; k < m; ++k) {
          dc[j][k] = (x[j] - z[j]) * y[1][k];
        }/* always */{
          dd[j]    = (x[j] - z[j]);
        }
      }

      double e0 = error(x, z);
      {
        // ランダムな微小変化配列を作る
        double[][] _ = new double[m][n];
        double[][] A = new double[m][n];
        for (int i = 0; i < m; ++i) {
          for (int j = 0; j < n; ++j) {
            _[i][j] = engine.nextDouble() * 1e-6;
            A[i][j] = a[i][j] + _[i][j];
          }
        }

        double e1 = error(x, cypd(c, axpb(A, x, b, unit)[1], d, type));

        // e0-e1 = trace[df(A)/dA^T _] となっているはず
        System.out.printf("A: %e\n", re(e0-e1, trace(mul(da, true, _))));
      }
      {
        // ランダムな微小変化配列を作る
        double[][] _ = new double[n][m];
        double[][] C = new double[n][m];
        for (int j = 0; j < n; ++j) {
          for (int i = 0; i < m; ++i) {
            _[j][i] = engine.nextDouble() * 1e-6;
            C[j][i] = c[j][i] + _[j][i];
          }
        }

        double e1 = error(x, cypd(C, y[1], d, type));

        // e0-e1 = trace[df(A)/dC^T _] となっているはず
        System.out.printf("C: %e\n", re(e0-e1, trace(mul(dc, true, _))));
      }
    }
  }

  // 相対誤差
  private static double re(double x, double z) {
    return (x - z) / x;
  }

  private static double trace(double[][] a) {
    assert(a.length == a[0].length);

    double retval = 0.0;
    for (int i = 0, size = a.length; i < size; ++i) {
      retval += a[i][i];
    }
    return retval;
  }

  void learn(int size, double[][] x, double alpha, Random engine) {
    // normalize
    double gamma = 1.0 - alpha * beta;
    if (gamma < 1.0) {
      for (int k = 0; k < m; ++k) {
        for (int j = 0; j < n; ++j) {
          a[k][j] *= gamma;
        }/* always */{
          b[k]    *= gamma;
        }
      }
      for (int j = 0; j < n; ++j) {
        for (int k = 0; k < m; ++k) {
          c[j][k] *= gamma;
        }/* always */{
          d[j]    *= gamma;
        }
      }
    }

    // calculate gradients:
    //   dL/dd =    x - z
    //   dL/dc = y (dL/dd)^T
    //   dL/db = A (dL/dd)
    //   dL/dA =   (dL/db) x^T
    for (int k = 0; k < m; ++k) {
      for (int j = 0; j < n; ++j) {
        da[k][j] = 0.0;
      }/* always */{
        db[k]    = 0.0;
      }
    }
    for (int j = 0; j < n; ++j) {
      for (int k = 0; k < m; ++k) {
        dc[j][k] = 0.0;
      }/* always */{
        dd[j]    = 0.0;
      }
    }

/*
    double[] w = new double[n];
    for (int i = 0; i < n; ++i) {
      w[i] = type.noise(x[i], engine);
    }
*/
    for (int i = 0; i < size; ++i) {
      double[][] y = axpb(a, x[i], b, unit);
      double[]   z = cypd(c, y[1], d, type);
      double[]   e = sub(x[i], z);

      //double[] atax = mul(a, true, mul(a, false, x   ));
      //double[] ctcy = mul(c, true, mul(c, false, y[1]));

      // Gradient:
      for (int k = 0; k < m; ++k) {
        double sum = 0.0;
        // C^T(x-z) * y * (1-y)
        for (int j = 0; j < n; ++j) {
          sum += c[j][k] * e[j];
        }
        sum *= unit.g(y[0][k], y[1][k]);

        for (int j = 0; j < n; ++j) {
          da[k][j] += sum * x[i][j];
        }/* always */{
          db[k]    += sum;
        }
      }
      for (int j = 0; j < n; ++j) {
        for (int k = 0; k < m; ++k) {
          dc[j][k] += e[j] * y[1][k];
        }/* always */{
          dd[j]    += e[j];
        }
      }
    }

    // update
    for (int k = 0; k < m; ++k) {
      for (int j = 0; j < n; ++j) {
        a[k][j] += alpha * da[k][j];
      }/* always */{
        b[k]    += alpha * db[k];
      }
    }
    for (int j = 0; j < n; ++j) {
      for (int k = 0; k < m; ++k) {
        c[j][k] += alpha * dc[j][k];
      }/* always */{
        d[j]    += alpha * dd[j];
      }
    }
  }

  private static void show(String name, double[] x) {
    System.out.printf("%s: ", name);
    for (double a : x) {
      System.out.printf("\t% 1.0f", a);
    }
    System.out.println();
  }

  private static void show(String name, double[][] a, double[] b) {
    System.out.printf("%s:\n", name);
    for (int i = 0, size = a.length; i < size; ++i) {
      for (double _ : a[i]) {
        System.out.printf("\t% 3.2f", _);
      }{
        System.out.printf("\t% 3.2f", b[i]);
      }{
        System.out.println();
      }
    }
    System.out.println();
  }

  private void show() {
    show("a", a, b);
    show("c", c, d);
  }

  private double error(double[] x, double[] z) {
    double retval = 0.0;
    {
      int n = x.length;
      for (int i = 0; i < n; ++i) {
        retval += type.error(x[i], z[i]);
      }
      //retval /= n;
    }
    return retval;
  }

  private void show(double[] x) {
    double[] y = axpb(a, x, b, unit)[1];
    double[] z = cypd(c, y, d, type);

    System.out.printf("%1.2e\n", error(x, z));

    // show log
    show("x", x);
    show("z", z);
    show("y", y);
  }

  public static void main(String[] args) {
/*
    double[][] x = new double[][] {
      {0, 1, 2, 3, 4, 3},
      {1, 2, 3, 2, 1, 0},
      {3, 2, 1, 0, 0, 1},
      {1, 0, 0, 1, 2, 3},
      {2, 1, 0, 1, 3, 2},
      {1, 2, 3, 2, 1, 0},
    };
/*/
    double[][] x = new double[][] {
      {1, 1, 0, 0, 0, 0},
      {0, 0, 1, 1, 0, 0},
      {0, 0, 0, 0, 1, 1},
      {1, 0, 0, 0, 0, 1},
      {1, 0, 1, 1, 0, 1},
      {1, 1, 0, 0, 1, 1},
      {1, 1, 0, 0, 1, 1},
      {1, 1, 1, 1, 0, 0},
      {0, 0, 1, 1, 1, 1},
    };
//*/
    int size   = x.length; // sample size
    int length = x[0].length; // the dimension of a vector

    Random engine = new Random();

    Autoencoder encoder = new Autoencoder(Type.BINOMIAL, Unit.SIGMOID, 0.0, length, length/2);

    for (int e = 1; e <= 100000; ++e) {
      encoder.learn(size, x, 1.0 / (Math.sqrt(e) * length), engine);
    }
    for (int i = 0; i < size; ++i) {
      encoder.show(x[i]);
    }
    encoder.show();

    for (int i = 0; i < 10; ++i) {
      encoder.test(engine);
    }
  }
}
