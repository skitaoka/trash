#include <vector>
#include <iostream>

typedef std::size_t (*hash_function)( char const * );

std::size_t hash1( char const * );
std::size_t hash2( char const * );
std::size_t hash3( char const * );

hash_function hash[] = { hash1, hash2, hash3 };
std::size_t const k = sizeof( hash ) / sizeof( hash[0] );
std::size_t const m = 32;
std::vector<bool> bit_vector( m, false );

std::size_t hash1( char const * w )
{
	std::size_t h = 0;
	for ( ; *w; ++w )
	{
		h = 5 * h + *w;
	}
	return h % m;
}

std::size_t hash2( char const * w )
{
	std::size_t h = 0;
	for ( ; *w; ++w )
	{
		h += *w;
	}
	return h % m;
}

std::size_t hash3( char const * w )
{
	std::size_t h = 0;
	for ( ; *w; ++w )
	{
		h ^= static_cast<std::size_t>( *w ) << 3;
	}
	return h % m;
}

void add_word( char const * const w )
{
	for ( std::size_t i = 0; i < k; ++i )
	{
		bit_vector[ hash[i]( w ) ] = true;
	}
}

bool is_containe( char const * const w )
{
	for ( std::size_t i = 0; i < k; ++i )
	{
		if ( !bit_vector[ hash[i]( w ) ] )
		{
			return false;
		}
	}

	return true;
}


int main( int argc, char * argv[] )
{
	if ( argc < 2 )
	{
		std::cerr << "Usage: " << argv[0] << " word" << std::endl;
		return 1;
	}

	char const * const input[] =
	{
		"foo",
		"bar",
		"baz",
		"quux",
/*
		"boo",
		"hoge",
		"fuga",
		"piyo",
		"hero",
		"foobar",
		"hogehoge",
*/
	};
	std::size_t const n = sizeof( input ) / sizeof( input[0] );

	for ( std::size_t i = 0; i < n; ++i )
	{
		add_word( input[i] );
	}

	if ( is_containe( argv[1] ) )
	{
		std::cout << "maybe" << std::endl;
	}
	else
	{
		std::cout << "no" << std::endl;
	}

	//
	// m/n = 8, k = 2 なので表(*)から擬陽性率は 0.0489
	//
	// 擬陽性: 存在しないのにあると判定されること
	// (*) false_positive_rate.xml
	//

	return 0;
}
