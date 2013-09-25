#include <iostream>
#include <vector>
#include <functional>
#include <algorithm>

class foo
{
public:
	foo( char const * name )
			: name_( name )
	{
	}

public:
	void bar( char const * name )
	{
		std::cout << name_ << ":" << name << std::endl;
	}

private:
	std::string const name_;
};


class baz
{
public:
	baz( char const * name )
			: name_( name )
	{
	}

public:
	void hoge( char const * name )
	{
		name_ += ":";
		name_ += name;
	}

	void hero()
	{
		std::cout << name_ << std::endl;
	}

private:
	std::string name_;
};

int main( void )
{
	std::vector<foo*> foos;
	foos.push_back( new foo( "aaa" ) );
	foos.push_back( new foo( "bbb" ) );
	foos.push_back( new foo( "ccc" ) );

	std::for_each( foos.begin(), foos.end(),
			std::bind2nd( std::mem_fun( &foo::bar ), "ddd" ) );

	std::vector<baz> bazs;
	bazs.push_back( baz( "aaa" ) );
	bazs.push_back( baz( "bbb" ) );
	bazs.push_back( baz( "ccc" ) );

	std::for_each( bazs.begin(), bazs.end(),
			std::bind2nd( std::mem_fun_ref( &baz::hoge ), "ddd" ) );

	std::for_each( bazs.begin(), bazs.end(),
			std::mem_fun_ref( &baz::hero ) );

	return 0;
}
