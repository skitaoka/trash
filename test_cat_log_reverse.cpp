#include <iostream>
#include <algorithm>
#include <iterator>
#include <string>
#include <vector>

/**
 * 一ラインごとに読み込んで逆順に出力する.
 */
int main()
{
	std::vector<std::string> lines;
	std::string temp;

	while ( !!std::cin )
	{
		std::getline( std::cin, temp );
		lines.push_back( temp );
	}

	std::copy( lines.rbegin(), lines.rend(),
			std::ostream_iterator<std::string>( std::cout, "\n" ) );
}
