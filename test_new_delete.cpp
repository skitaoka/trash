#include <iostream>
#include <memory>

class foo
{
public:
	static void * operator new ( std::size_t size )
	{
		void * p = ::operator new ( size );
		std::cout << p << " = operator new ( " << size << " )" <<  std::endl;
		return p;
	}

	static void operator delete ( void * p )
	{
		std::cout << "operator delete ( " << p << " )" << std::endl;
		::operator delete ( p );
	}

	static void * operator new [] ( std::size_t size )
	{
		return operator new ( size );
	}

	static void operator delete [] ( void * p )
	{
		operator delete ( p );
	}

public:
	foo()
		: bar_( new int( 0 ) )
		, baz_( new int[ 10 ] )
	{
	}

	~foo()
	{
		delete bar_;
		delete [] baz_;
	}

private:
	int * bar_;
	int * baz_;
};

int main( int argc, char * argv [] )
{
	std::auto_ptr<foo> hoge( new foo() );

	foo * hero = new foo[ 10 ];
	delete [] hero;

	return EXIT_SUCCESS;
}
