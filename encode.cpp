#include <stdio.h>
#include <stdlib.h>

#define N (4*1024*1024)
char bits[N];
char buf[N];

/// 適当暗号化
int main( int argc, char * argv[] )
{
  FILE * in = NULL;
  FILE * out = NULL;

  if ( argc < 2 )
  {
    goto EXIT;
  }

  unsigned int key = ('t') | ('e'<<8) | ('s'<<16) | ('t'<<24);
  ::srand( key );

  for ( size_t i = 0; i < N; ++i )
  {
    bits[i] = static_cast<char>( (::rand()>>4)&0xff );
  }

  for ( int i = 1; i < argc; ++i )
  {
    if ( ( in = ::fopen( argv[i], "rb" ) ) == NULL )
    {
      goto EXIT;
    }

    char filename[256];
    ::sprintf( filename, "%s.tmp", argv[i] );
    if ( ( out = ::fopen( filename, "wb" ) ) == NULL )
    {
      goto EXIT;
    }

    size_t n;
    while ( ( n = ::fread( buf, 1, N, in ) ) > 0 )
    {
      for ( size_t i = 0; i < n; ++i )
      {
        buf[i] ^= bits[i];
      }
      ::fwrite( buf, 1, n, out );
    }

  EXIT:
    if ( in )
    {
      ::fclose( in );
    }
    if ( out )
    {
      ::fclose( out );
    }
  }

  return 0;
}
