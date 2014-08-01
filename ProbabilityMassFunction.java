// x = 0...(n-1) の離散値をとる確率質量関数の定義式を考えた
class ProbabilityMassFunction {

  // 内部パラメータ
  private static double[] p = {0.1, 0.4, 0.2, 0.3};

  // x = 0...(n-1) の離散値をとる確率質量関数
  private static double prob(int x) {
    // return p[x] と同値

    double retval = 1.0;
    for (int i = 0, length = p.length; i < length; ++i) {
      double param = 1.0;
      for (int j = 0; j < length; ++j) {
        if (i != j) { param *= (double)(j - x) / (j - i); }
        //if (x != j) { param *= (double)(j - i) / (j - x); }
      }
      System.out.println(param);
      retval *= Math.pow(p[i], param);
    }
    return retval;
  }

  public static void main(String[] args) {
    for (int i = 0, length = p.length; i < length; ++i) {
      prob(i);
      //System.out.println(prob(i));
    }
  }
}
