class Radiosity {

public:
  // Gauss-Seidel法
  void doMatrixMethod(Patch* aPatch, int nPatch,
          double** aadCoefficient, int nStep) {
    for (int i = 0; i < nPatch; i++) {
      aPatch[i].dRadiance = 0;
    }

    // 適当な回数繰り返す。
    for (int n = 0; n < nStep; n++) {
      // 全てのパッチに対して
      for (int i = 0; i < nPatch; i++) {
        // i 行の方程式の自分を除いたのを得る。
        // （内積をとっている）
        double bk = 0;
        for (int j = 0 ; j < nPatch; j++) {
          if (i != j) {
            bk += aadCoefficient[i][j] * aPatch[j].dRadiance;
          }
        }

        // ずれを直す
        aPatch[i].dRadiance = aPatch[i].dEmission - bk;
      }
    }
  }

  // Southwell漸進法
  void doProgressiveRefinement(Patch* aPatch, int nPatch,
          double** aadCoefficient, int nStep) {
    boost::shard_array adRadiance(new double[nPatch]);

    // 自己発光を覚えとく
    for (int i = 0; i < nPatch; i++) {
      adRadiance[i] = aPatch[i].dEmission;
    }

    // 適当な回数繰り返す。
    for (int i = 0; i < nStep; i++) {

      // 最大の放射発散度を持つ項を選択
      int bi = 0;
      double bk = adRadiance[0];
      for (int j = 1; j < nPatch; j++) {
        if (bk < adRadiance[j]) {
          bk = adRadiance[j];
          bi = j;
        }
      }

      // 全てのパッチに最大の放射発散度を割り当てる
      for (int j = 0; j < nPatch; j++) {
        double dk = aadCoefficient[bi][j] * adRadiance[bi];
        adRadiance[j] -= dk;
        patch[j].dRadiance -= dk;
      }

      adRadiance[bi] = 0;
    }
  }
}
