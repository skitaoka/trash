// ΔΣ変調
final class DeltaSigmaModulation {
  public static void main(String[] args) {
    final int nFinalResolution = 256;
    final int nSuperSamplingRate = 16;

    double[] noise = new double[nFinalResolution]; // 量子化ノイズ
    for (int i = 0; i < nFinalResolution; ++i) {
      noise[i] = 2 * Math.random() - 1;
    }

    double[] fx_a = new double[nFinalResolution];
    double sum_fx_a = 0;
    for (int i = 0; i < nFinalResolution; ++i) {
      double x = Math.PI * i / nFinalResolution;
      fx_a[i] = 255 * Math.cos(x);
      sum_fx_a += fx_a[i];
    }

    // 通常の量子化
    double[] fx_n = new double[nFinalResolution];
    double sum_fx_n = 0;
    for (int i = 0; i < nFinalResolution; ++i)　{
      fx_n[i]  = (int)Math.round( fx_a[i] + noise[i] );
      sum_fx_n += fx_n[i];
    }
    System.out.println(sum_fx_n - sum_fx_a);

    // ΔΣ変調を使った量子化
    double[] fx_m = new double[nFinalResolution];
    double sum_fx_m = 0;
    {
//*
      double delta = 0;
      for (int i = 0; i < nFinalResolution; ++i) {
        fx_m[i] = (int)Math.round(fx_a[i] + delta + noise[i]);
        delta = fx_a[i] - fx_m[i];
        sum_fx_m += fx_m[i];
      }
/*/
      // まず 4 bit に量子化する
      int[] fx_s = new int[nFinalResolution * nSuperSamplingRate];
      double delta = 0;
      for (int i = 0, size = fx_s.length; i < size; ++i) {
//        double f = fx_a[i/nSuperSamplingRate] / nSuperSamplingRate;
        double r = Math.random(); // ノイズを付加
        double f = fx_a[i/nSuperSamplingRate] + r;
        fx_s[i] = (int)Math.round(f + delta);
        delta = f - fx_s[i] - r;
      }
      
      // 平均化する
      for (int i = 0; i < nFinalResolution; ++i) {
        int sum = 0;
        for (int j = 0; j < nSuperSamplingRate; ++j) {
          sum += fx_s[nSuperSamplingRate * i + j];
        }
        fx_m[i] = (int)Math.round((double)sum / nSuperSamplingRate);
      }
//*/
    }
    System.out.println( sum_fx_m - sum_fx_a );

/*
    System.out.printf(",A,B,C\n");
    for (int i = 0; i < nFinalResolution; ++i) {
      System.out.printf("%d,%f,%f,%f\n", i, fx_a[i], fx_n[i], fx_m[i]);
    }
*/
  }
}
