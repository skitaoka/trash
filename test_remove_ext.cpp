#include <iostream>
#include <memory>

inline std::auto_ptr<std::string> remove_ext( std::string& filename )
{
	return std::auto_ptr<std::string>( new std::string( filename, 0, filename.find_first_of( ".txt" ) ) );
}

int main( void )
{
	std::string filename( "hogehoge.txt" );
	std::auto_ptr<std::string> nonext( remove_ext( filename ) );

	std::cout << filename << std::endl;
	std::cout << *nonext << std::endl;

	return 0;
}
