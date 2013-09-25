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
	std::string line;

	while (std::cin) {
		std::getline(std::cin, line);
		lines.push_back(line);
	}

	std::copy(lines.rbegin(), lines.rend(),
			std::ostream_iterator<std::string>(std::cout, "\n"));
}
