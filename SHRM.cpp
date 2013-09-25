/*
          ┌                    ┐  ┌                            ┐
          │  R_yy  R_yz  R_yx  │  │  R_-1,-1  R_-1,0  R_-1,+1  │
    R_mn＝│  R_zy  R_zz  R_zx  │＝│  R_ 0,-1  R_ 0,0  R_ 0,+1  │
          │  R_xy  R_xz  R_xx  │  │  R_+1,-1  R_+1,0  R_+1,+1  │
          └                    ┘  └                            ┘

The SH rotation matrix M is defined for band l as
    M_mn_{l}＝u_mn_{l} * U_mn_{l} +
              v_mn_{l} * V_mn_{l} +
              w_mn_{l} * W_mn_{l}


初期状態
          ┌                       ┐
          │  1  0     0     0     │
    R_mn＝│  0  R_yy  R_yz  R_yx  │
          │  0  R_zy  R_zz  R_zx  │
          │  0  R_xy  R_xz  R_xx  │
          └                       ┘
*/

void uvw(double& u, double& v, double& w, int m, int n, int l)
{
  double sigma_m0 = sigma(m, 0);
  double abs_m = abs(m);

  if ((n == l) || (-n == l)) {
    double d = 1 / ((2 * l) * (2 * l - 1));
    u =                               sqrt(d * (l + m) * (l - m));
    v =  0.5 * (1 - 2 * sigma_m0) * sqrt(d * (l + abs_m - 1) * (l + abs_m) * (1 + sigma_m0));
    w = -0.5 * (1 -     sigma_m0) * sqrt(d * (l + abs_m - 1) * (l + abs_m));
  }

  { // |n| < l
    double d = 1 / ((l + n) * (l - n));
    u =                               sqrt(d * (l + m) * (l - m));
    v =  0.5 * (1 - 2 * sigma_m0) * sqrt(d * (l + abs_m - 1) * (l + abs_m) * (1 + sigma_m0));
    w = -0.5 * (1 -     sigma_m0) * sqrt(d * (l - abs_m - 1) * (l - abs_m));
  }
}

double U(int m, int n, int l)
{
  return P(0, m, n, l);
}

double V(int m, int n, int l)
{
  if (m > 0) {
    return  P(1,  m - 1, n, l) * sqrt(1 + sigma(m,  1)) -
        P(-1, -m + 1, n, l) *     (1 - sigma(m,  1));
  }

  if (m < 0) {
    return  P(1,  m + 1, n, l) *     (1 + sigma(m, -1)) +
        P(-1, -m - 1, n, l) * sqrt(1 - sigma(m, -1));
  }

  { // m = 0
    return  P(1, 1, n, l) + P(-1, -1, n, l);
  }
}

double W(int m, int n, int l)
{
  if (m > 0) {
    return P(1, m + 1, n, l) + P(-1, -m - 1, n, l);
  }

  if (m < 0) {
    return P(1, m - 1, n, l) + P(-1, -m + 1, n, l);
  }

  { // m = 0
    assert(!"ここに来たら駄目");
    return 0;
  }
}

double P(int i, int a, int b, int l)
{
  if (b ==  l) {
    return  R(i,  1) * M(a,  l - 1, l - 1) -
        R(i, -1) * M(a, -l + 1, l - 1);
  }

  if (b == -l) {
    return  R(i,  1) * M(a, -l + 1, l - 1) +
        R(i, -1) * M(a,  l - 1, l - 1);
  }

  { // |b| < l
    return  R(i, 0) * M(a, b, l - 1);
  }
}

double sigma(int m, int n)
{
  if (m == n) {
    return 1;
  }

  {
    return 0;
  }
}

double M(int m, int n, int l)
{
  double u;
  double v;
  double w;

  uvw(u, v, w, m, n, l);

  return u * U(m, n, l) +
         v * V(m, n, l) +
         w * W(m, n, l);
}
