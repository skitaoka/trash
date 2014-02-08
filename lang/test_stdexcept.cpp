#include <iostream>
#include <stdexcept>

int main( int argc, char* argv )
{
	std::runtime_error e1( "hoge" );
	std::runtime_error e2( e1 );

	std::cout << e1.what() << std::endl;
	std::cout << e2.what() << std::endl;

	if ( e1.what() == e2.what() )
	{
		std::cout << "浅いコピー" << std::endl;
	}
	else
	{
		std::cout << "深いコピー" << std::endl;
	}

	return 0;
}
