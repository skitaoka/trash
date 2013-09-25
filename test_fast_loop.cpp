#include <stdio.h>
#include <stdlib.h>

void send_a( register double * to, register const double * from, register int count )
{
	register int n = ( count + 7 ) / 8;
	switch ( count & 3 )
	{
	case 0: do {
	case 7:   *to++ *= *from++;
	case 6:   *to++ *= *from++;
	case 5:   *to++ *= *from++;
	case 4:   *to++ *= *from++;
	case 3:   *to++ *= *from++;
	case 2:   *to++ *= *from++;
	case 1:   *to++ *= *from++;
			} while ( --n );
	}
}

void send_b( register double * to, register const double * from, register int count )
{
	while ( count-- )
	{
		*to++ *= *from++;
	}
}

int main( void )
{
	int n = 100000;
	int count = 100 * 100 * 100;
	double * from = (double *)malloc( sizeof( double ) * count );
	double * to = (double *)malloc( sizeof( double ) * count );

	while ( n-- )
	{
		//send_a( to, from, count );
		send_b( to, from, count );
	}

	printf( "%d\n", to[0] );

	return EXIT_SUCCESS;
}
