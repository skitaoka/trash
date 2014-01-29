#include <iostream>

template <typename T1, typename T2>
struct Foo
{
	void print()
	{
		std::cout << "this is T1 T2 structure." << std::endl;
	}
};

template <typename T1>
struct Foo<T1, float>
{
	void print()
	{
		std::cout << "this is T1 float structure." << std::endl;
	}
};

template <typename T2>
struct Foo<float, T2>
{
	void print()
	{
		std::cout << "this is float T2 structure." << std::endl;
	}
};

template <>
struct Foo<int, int>
{
	void print()
	{
		std::cout << "this is int int structure." << std::endl;
	}
};

template <typename T1, typename T2>
void func( T1, T2 )
{
	std::cout << "this is T1 T2 function." << std::endl;
}

template <typename T1>
void func( T1, float )
{
	std::cout << "this is T1 float function." << std::endl;
}

template <typename T2>
void func( float, T2 )
{
	std::cout << "this is float T2 function." << std::endl;
}

template <>
void func( int, int )
{
	std::cout << "this is int int function." << std::endl;
}

int main( void )
{

	Foo<double, double> a;
	Foo<double, float > b;
	Foo<float , double> c;
	Foo<int   , int   > d;

	a.print(); //==> this is T1 T2 structure.
	b.print(); //==> this is T1 float structure.
	c.print(); //==> this is float T2 structure.
	d.print(); //==> this is int int structure.

	func( 1.0 , 2.0  ); //==> this is T1 T2 function.
	func( 1.0 , 2.0f ); //==> this is T1 float function.
	func( 1.0f, 2.0  ); //==> this is float T2 function.
	func( 1   , 2    ); //==> this is int int function.

	return 0;
}
