#include <iostream>

template <std::size_t Sz>
union hoge
{
	int a;
	int b;
};

template<>
union hoge<2>
{
	int c;
	int d;
};

int main( int argc, char* argv[] )
{
	hoge<2> a;
	a.c = 1;
	a.d = 2;

	std::cout << a.c << std::endl;
	std::cout << a.d << std::endl;

	return 0;
}
