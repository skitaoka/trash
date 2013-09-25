// Restricted Boltmann Machine
// cf. http://en.wikipedia.org/wiki/Restricted_Boltzmann_machine

// これを多段に拡張すると Deep Learning できる！
// 特徴選択、普遍化、正規化の 3 Layer は必要？

size_t n; // 学習用の入力データ数
size_t l; // 入力ベクトルの次元数
size_t m; // 出力ベクトルの次元数

double v[n][l+1]; // 可視変数 = 入力(0-1)
double h[n][m+1]; // 隠れ変数
double x[n][l+1]; // 可視変数の中間状態
double y[n][m+1]; // 隠れ変数の中間状態

// 最適化するパラメータ
double w[(m+1)*(l+1)]; // 重み
double d[(m+1)*(l+1)]; // 勾配

// 学習の収束パラメータ
size_t count = 100; // 反復パラメータ
double alpha = 0.1; // 勾配の重み

void learn()
{
  //todo: v にデータを読み込む

  for (size_t k = 0; k < n; ++k) {
    v[k][l] = 1.0;
    h[k][m] = 1.0;
    x[k][l] = 1.0;
    y[k][m] = 1.0;
  }

  for (size_t loop = 0; loop < count; ++loop) { // 適当な回数反復する
    // Gibbs Sampling
    for (size_t k = 0; k < n; ++k) {
      sample_phv(h[k], m, v[k], l, w); // p(h|v)
      sample_pvh(x[k], l, h[k], m, w); // p(x|v) ~ p(v|h)
      sample_phv(y[k], m, x[k], l, w); // p(y|x) ~ p(h|v)
    }

    // 勾配
    for (size_t j = 0; j <= m; ++j) {
      for (size_t i = 0; i <= l; ++i) {
        double retval = 0.0;
        for (size_t k = 0; k < n; ++k) {
          retval += v[k][i] * h[k][j]; // positive gradient
          retval -= x[k][i] * y[k][j]; // negative gradient
        }
        d[j*(l+1)+i] = retval / n;
      }
    }

    // 更新
    for (size_t j = 0; j <= m; ++j) {
      for (size_t i = 0; i <= l; ++i) {
        w[j*(l+1)+i] += alpha * d[i*l+j];
      }
    }

    // note: 最後の要素は計算に関係しない。
    d[m*(l+1)+l] = 0.0;
    w[m*(l+1)+l] = 0.0;
  }

  // todo: w を書き出す (識別に使うだけなら最後の行はいらない)。
}

// v ~ P(v|h)
void sample_pvh(
    double       v[], size_t const l,
    double const h[], size_t const m,
    double const w[])
{
  for (size_t i = 0; i < l; ++i) {
    double retval = 0.0;
    for (size_t j = 0; j <= m; ++j) {
      retval += w[j*(l+1)+i] * h[j];
    }
    v[i] = sigmoid(retval);
  }
  //h[l] = 1.0; // 初期化しているならこの処理はいらない
}

// h ~ P(h|v)
void sample_phv(
    double       h[], size_t const m,
    double const v[], size_t const l,
    double const w[])
{
  for (size_t j = 0; j < m; ++j) {
    double retval = 0.0;
    for (size_t i = 0; i <= l; ++i) {
      retval += w[j*(l+1)+i] * v[i];
    }
    h[j] = sigmoid(retval);
  }
  //h[m] = 1.0; // 初期化しているならこの処理はいらない
}

double sigmoid(double const x)
{
  return 1 / (1 + std::exp(-x));
}

/*
可視層(v={v_i})と隠れ層(h={h_j})をもつボルツマンマシンを考える。
可視層と隠れ層のあいだにリンクがある構造で、可視層間や隠れ層間にはリンクがない構造にする。
つまり P(h|v) = PI_j P(h_j|v) として h_i が独立になるモデルにする。
具体的には、
	P(v_i=1|h) = sigmoid(SUM_j w_ij h_j)	... (1)
	P(h_j=1|v) = sigmoid(SUM_i w_ij v_i)	... (2)
	sigmoid(x) = 1/{1+exp(-x)}
とする。

モデルを
	p(v) = SUM_h p(v,h)
	p(v,h) = exp(-E(v,h))/Z
	E(v,h) = - SUM_i SUM_j v_i w_ij h_j
	Z = SUM_v SUM_h exp{-E(v,h)}
とする。

可視ベクトルの集合 V={v^1..v^n} が与えられている条件で、
それを上記のモデルで最も尤もらしく表現する
重み行列 W={w_ij} を学習する。

最尤推定する。
	argmax_{W,b,c} PI_v p(v)
対数尤度を最大化する問題に書きかえる。
	argmax_{W,b,c} SIGMA_v log(p(v))    ... (3)
ここで、あたえられた入力の集合が確率分布
	q(v) = 1/n SIGMA_k delta(v-v^k) に
に従っている考えられるので eq.3 は期待値計算をしていることになる。
そこで、
	argmax_{W,b,c} <log(p(v))>_q(v)
と書きかえる。
	J = <log(p(v))>_q(v)
とおいて、式変形すると、
	  = <log[SUM_h p(v,h)]>_q(v)
	  = <log[SUM_h exp{-E(v,h)}/Z]>_q(v)
	  = <log[SUM_h exp{-E(v,h)}] - log(Z)>_q(v)
	  = <log[SUM_h exp{-E(v,h)}]>_q(v) - log(Z)
となる。こいつの勾配を求める。
	dJ/w_ij = d/dw_ij [<log[SUM_h exp{-E(v,h)}]>_q(v) - log(Z)]
	        = d/dw_ij [<log[SUM_h exp{-E(v,h)}]>_q(v)] - 1/Z dZ/dw_ij
	        = d/dw_ij [<log[SUM_h exp{-E(v,h)}]>_q(v)] + 1/Z d/dw_ij[SUM_v SUM_h dE(v,h)/dw_ij exp{-E(v,h)}]    ... Z = SUM_v SUM_h exp{-E(v,h)}
	        = d/dw_ij [<log[SUM_h exp{-E(v,h)}]>_q(v)] + SUM_v SUM_h dE(v,h)/dw_ij p(v,h)                       ... p(v,h) = exp{-E(v,h)} / Z
	        = d/dw_ij [<log[SUM_h exp{-E(v,h)}]>_q(v)] + <dE(v,h)/dw_ij>_p(v,h)
	        = -<[SUM_h dE(v,h)/dw_ij exp{-E(v,h)}]/[SUM_h exp{-E(v,h)}]>_q(v) + <dE(v,h)/dw_ij>_p(v,h)
	        = -<SUM_h dE(v,h)/dw_ij p(h|v)>_q(v) + <dE(v,h)/dw_ij>_p(v,h)                                       ... p(h|v) = exp{-E(v,h)} / [SUM_h exp{-E(v,h)}]
	        = -<dE(v,h)/dw_ij>_p(h|v)q(v) + <dE(v,h)/dw_ij>_p(v,h)
	        = <v_i h_j>_p(h|v)q(v) - <v_i h_j>_p(v,h)

p(h|v)q(v) と p(v,h) から、それぞれサンプリングして勾配を求める。

p(h|v)q(v) からのサンプリング(直接計算できる): positive gradient
	eq.2 で {v^1...v^n} から {h^1...h^n} を計算する。
	<v_i h_j>_p(h|v)q(v) = 1/n SUM_k v_i^k h_j^k とする。

p(v,h) からのサンプリング: negative gradient
	p(h|v)q(v) から Gibbs sampling する。
	eq.1 で {h^1...h^n} から {x^1...x^n} を計算する(ここが Gibbs sampling)。
	eq.2 で {x^1...x^n} から {y^1...y^n} を計算する(直接計算)。
	<v_i h_j>_p(v,h) = 1/n SUM_k x_i^k y_j^k とする。
*/
