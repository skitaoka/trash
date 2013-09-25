#include <stdio.h>

void func( int a )
{
	printf( "func called %d\n", a );
}

int main( void )
{
	typedef void (*pFuncType)( int );

	pFuncType pFunc = func;

	printf( "&func: %x\n", &func );
	printf( " func: %x\n",  func );
	printf( "*func: %x\n", *func );

	printf( "&pFunc: %x\n", &pFunc );
	printf( " pFunc: %x\n",  pFunc );
	printf( "*pFunc: %x\n", *pFunc );

	(&func)( 1 );
	func( 2 );
	(*func)( 3 );

	//(&pFunc)( 4 );
	pFunc( 5 );
	(*pFunc)( 6 );

	return 0;
}
