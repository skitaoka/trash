#include <iostream>

struct a
{
	a() throw()
	{
		std::cout << "a()" << std::endl;
	}
	a( const a& ) throw()
	{
		std::cout << "a( const a& )" << std::endl;
	}
	virtual ~a() throw() {}
};

struct b : public a
{
	b() throw()
	{
		std::cout << "b()" << std::endl;
	}
};

int main( int argc, char* argv )
{
	b hoge;
	b hero( hoge );

	return 0;
}
