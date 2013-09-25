#include <iostream>
#include <cmath>

inline double square(double const x)
{
  return x * x;
}

class matrix_t
{
public:
  inline matrix_t(double const _a, double const _b,
                  double const _c, double const _d)
    : a(_a), b(_b), c(_c), d(_d)
  {
  }

  inline matrix_t operator * (matrix_t const & m) const
  {
    return matrix_t(
      a * m.a + b * m.c,
      a * m.b + b * m.d,
      c * m.a + d * m.c,
      c * m.b + d * m.d);
  }

  inline double get_reflectance(double const rho) const
  {
    return (c - rho * a) / (rho * b - d);
  }

private:
  double a;
  double b;
  double c;
  double d;
};

matrix_t get_transfer_matrix(double const sigma_s, double const sigma_a, double const z0)
{
  double const sigma_t = sigma_s + sigma_a;
  double const a = sigma_t / sigma_s;
  double const b = std::sqrt(square(a) - 1.0);
  double const ep = std::exp( b * sigma_s * z0);
  double const em = std::exp(-b * sigma_s * z0);
  double const sinh = (ep - em) * 0.5;
  double const cosh = (ep + em) * 0.5;

  double const k = a * sinh + b * cosh;
  double const c = 1.0 / b;

  return matrix_t((square(b) - square(sinh)) * c / k, sinh * c, -sinh * c, k * c);
}

int main()
{
  int    const n       = 10000;
  double const sigma_s = 0.3;
  double const sigma_a = 0.3;
  double const z0      = 0.01;
  double const rho     = 0.9;

  matrix_t r(1.0, 0.0, 0.0, 1.0);
  for (int i = 0; i < n; ++i) {
    r = get_transfer_matrix(sigma_s, sigma_a, z0) * r;
  }

  std::cout << "m: " << r.get_reflectance(rho) << std::endl;

  r = get_transfer_matrix(sigma_s, sigma_a, z0 * n);
  std::cout << "d: " << r.get_reflectance(rho) << std::endl;

  return 0;
}
/*
[単層の場合]

垂直方向の入射光と出射光を考える。
      0 |  ↓ i+( 0)  ↑ i-( 0)
--------+-----------------------
        |  ↓ i+ 
    ----+------------------- dz
    ----+-------------------
        |            ↑ i-
--------+-----------------------
      z0|  ↓ i+(z0)  ↑ i-(z0)

  i+( 0) = i0   ... 入射面 z= 0 で i+ は入射光 i0 に一致する
  i-(z0) =  0　  ... 出射面 z=z0 で i- は反射光  0 に一致する

d  |i+|   |-σ_t σ_s||i+|
-- |  | = |        ||  |
dz |i-|   |-σ_s σ_t||i-|

   d
⇔ -- I = Q . I
   dz

  σ_s: 散乱率
  σ_a: 吸収率
  σ_t = σ_a + σ_s: 減衰率

Q を固有値分解してみる。

V^-1 Q V = Λ

    | v11 v21 |
V = |         |
    | v21 v22 |

    | λ_1  0  |
L = |         |
    |  0  λ_2 |

where
  λ_1,2 = ±√(σ_t^2 + σ_s^2)

                 1        | 1 |
  v_1,2 = --------------- |   |
           √(2a^2 ± 2ab)  |a±b|

          |        1               1      |
          |--------------- ---------------|
          | √(2a^2 + 2ab)   √(2a^2 - 2ab) |
        = |                               |
          |      a + b           a - b    |
          |--------------- ---------------|
          | √(2a^2 + 2ab)   √(2a^2 - 2ab) |
  and
         σ_t
    a = -----
         σ_s

    b = √(a^2 - 1)

I~ = V^-1 I とおいてみる。

     d
     -- I = Q . I
     dz

     d
  ⇔ -- I = (V Λ V^-1) I
     dz
   
     d
  ⇔ -- V^-1 I = Λ V^-1 I
     dz

     d
  ⇔ -- I~ = Λ I~
     dz

一階線型常微分方程式なので、

       | c1   0 || exp(λ1 z) |
  I~ = |        ||           |
       |  0  c2 || exp(λ2 z) |

       | c1 exp(λ1 z) |
     = |              |
       | c2 exp(λ2 z) |

が得られる。よって、

  I = V I~

      | v11 v21 || c1 exp(λ1 z) |
    = |         ||              |
      | v21 v22 || c2 exp(λ2 z) |

となり、c1 と c2 を求めるために境界条件を与えると、

          | v11  v21 || c1           |   | i0 | ... (*)
  I( 0) = |          ||              | = |    |
          | v12  v22 || c2           |   | x  |

          | v11  v21 || c1 exp(λ1 z) |   | y  |
  I(z0) = |          ||              | = |    |
          | v12  v22 || c2 exp(λ2 z) |   |  0 | ... (**)

となる。

ここで、(*) と (**) を連立させると

        | i0 |
  F c = |    |
        |  0 |

      | v11            v21           |
  F = |                              |
      | v12 exp(λ1 z)  v22 exp(λ2 z) |

      |　c1 |
  c = |   　|
      | c2 |

           | i0 |
    = F^-1 |    |
           |  0 |

となる。

がんばって整理する。

  I = V I~

      | v11 v21 || exp(λ1 z)           || c1 |
    = |         ||                     ||    |
      | v21 v22 ||     0     exp(λ2 z) || c2 |

      | v11 v21 || exp(λ1 z)           |      | i0 |
    = |         ||                     | F^-1 |    |
      | v21 v22 ||     0     exp(λ2 z) |      |  0 |

    (中略)


                        i0                  | a sinh[b σ_s (z0-z)] + b cosh[b σ_s (z0-z)] |
    = ------------------------------------- |                                             |
       a sinh[b σ_s z0] + b cosh[b σ_s z0]  |   sinh[b σ_s (z0-z)]                        |

i0 = i+(0) = 1 として、反射率と透過率を求めると、

  | T |   | i+(z0) |
  |   | = |        |
  | R |   | i-( 0) |

                            1                   | a sinh[b σ_s (z0-z0)] + b cosh[b σ_s (z0-z0)] |   ... z=z0
        = ------------------------------------- |                                               |
           a sinh[b σ_s z0] + b cosh[b σ_s z0]  |   sinh[b σ_s (z0- 0)]                         |   ... z= 0


                            1                   |          b          |
        = ------------------------------------- |                     |
           a sinh[b σ_s z0] + b cosh[b σ_s z0]  | sinh[b σ_s (z0- 0)] |

    where
      T: 透過率
      R: 反射率

が得られる。

[複層の場合]

複数層が重なっている状態を考える。

たとえば3層の場合、

  I2 = A2 I1
     = A2 A1 I0

となるような Ai を求める。

  | i+(j+1) |   | T  R || i+(j  ) |   ... (1)
  |         | = |      ||         |
  | i-(j  ) |   | R  T || i-(j+1) |   ... (2)

(2) より

     i-(j) = R i+(j) + T i-(j+1)
  ⇔
    T i-(j+1) = i-(j) - R i+(j)
  ⇔
                i-(j) - R i+(j)
    i-(j+1) = -----------------
                      T
  ⇔
              | -R    1  || i+(j) |
    i-(j+1) = | ---  --- ||       |    ... (*)
              |  T    T  || i-(j) |


(*) を (1) に代入する。
                         i-(j) - R i+(j)
  i+(j+1) = T i+(j) + R -----------------
                               T

            |      R^2    R  || i+(j) |
          = | T - -----  --- ||       |
            |       T     T  || i-(j) |

まとめると
  | i+(j+1) |    1  | 1-R^2  R || i+(j) |
  |         | = --- |          ||       |
  | i-(j+1) |    T  |  -R    1 || i-(j) |

          1  | 1-R^2  R |
    Ai = --- |          |
          T  |  -R    1 |

[備考]

複数層重なったときの反射光を求める。

  A = An ... Ai ... A1

      | a  b |
    = |      |
      | c  d |

  | i+(n) |   | a  b || i+(0) |
  |       | = |      ||       |
  | i-(n) |   | c  d || i-(0) |

入射光を Ii とすると   : i+(0)　= Ii
紙に届く光を X とすると : i+(n) = X
紙の反射率を ρ とすると: i-(n) = ρ i+(n) = ρ X

反射光 Io = i-(0) は
      X = a Ii + b Io
    ρ X = c Ii + d Io
  ⇔
    ρ (a Ii + b Io) = c Ii + d Io
  ⇔
    ρ a Ii + ρ b Io = c Ii + d Io
  ⇔
    ρ b Io - d Io = c Ii - ρ a Ii
  ⇔
          c - ρ a 
    Io = --------- Ii
          ρ b - d
で求められる

反射率 R は、 Ii = 1 として

       c - ρ a 
  R = ---------
       ρ b - d

で得られる。

*/
