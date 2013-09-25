#include <stdio.h>
#include <math.h>

struct float_int
{
  union
  {
    unsigned i;
    float    f;
  };
};

void float2rgbe( float m )
{
  int n;
  printf("m = %f  ", frexp(m, &n));
  printf("n = %d\n", n + 128);
}

void _my_float2rgbe(float const m)
{
  float_int fi;
  fi.f = m;

  int const n = ((fi.i & 0x7F800000) >> 23) + 2;
  if (fi.i & 0x80000000) {
    m = -((fi.i & 0x007FFFFF) * (1.0f / 0x01000000) + 0.5f);
  } else {
    m =  ((fi.i & 0x007FFFFF) * (1.0f / 0x01000000) + 0.5f);
  }

  printf( "m = %f  ", m );
  printf( "n = %d\n", n );
}

void rgbe2float(float const f, int const n)
{
  printf( "x = %f\n", f * ldexp(1.0, n - (128 + 8)));
}

void _my_rgbe2float(float const f, int const n)
{
  float const m = (float)pow(2, n - ( 128 + 8 ));
  printf( "x = %f\n", m * f );
}

int main()
{
  float v = 0.513234f;

      float2rgbe(v);
  _my_float2rgbe(v);

  int n = 145;

      rgbe2float(v, n);
  _my_rgbe2float(v, n);

  return 0;
}
