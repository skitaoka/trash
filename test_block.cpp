#include <cassert>
#include <iostream>
#include <iterator>
#include <algorithm>

/**
 * 非初期化型
 * 代入型
 */
template < typename T, std::size_t Sz >
struct block
{
public:
	typedef T value_type;
	typedef value_type * pointer;
	typedef value_type const * const_pointer;
	typedef value_type & reference;
	typedef value_type const & const_reference;

	typedef std::ptrdiff_t difference_type;
	typedef std::size_t size_type;

	typedef pointer iterator;
	typedef const_pointer const_iterator;

	typedef std::reverse_iterator<iterator> reverse_iterator;
	typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

public:
	value_type data_[Sz];

public:
	iterator begin() { return data_; }
	iterator end() { return data_ + Sz; }

	const_iterator begin() const { return data_; }
	const_iterator end() const { return data_ + Sz; }

	reverse_iterator rbegin() { return reverse_iterator( end() ); }
	reverse_iterator rend() { return reverse_iterator( begin() ); }

	const_reverse_iterator rbegin() const { return const_reverse_iterator( end() ); }
	const_reverse_iterator rend() const { return const_reverse_iterator( begin() ); }

public:
	reference operator [] ( size_type i )
	{
		assert( ( 0 <= i ) && ( i < Sz ) );
		return data_[i];
	}

	const_reference operator [] ( size_type i ) const
	{ 
		assert( ( 0 <= i ) && ( i < Sz ) );
		return data_[i];
	}

public:
	size_type size() const { return Sz; }
	size_type max_size() const { return Sz; }
	bool empty() const { return Sz == 0; }

public:
	void swap( block& x )
	{
		for ( size_type i = 0; i < Sz; ++i )
		{
			std::swap( data_[i], x.data_[i] );
		}
	}
};

/**
 * 等式比較型
 */
template <typename T, std::size_t Sz>
bool operator == ( block<T, Sz> const & lhs, block<T, Sz> const & rhs )
{
	for ( std::size_t i = 0; i < Sz; ++i )
	{
		if ( lhs.data_[i] != rhs.data_[i] )
		{
			return false;
		}
	}

	return true;
}

/**
 * 小なり比較型
 */
template <typename T, std::size_t Sz>
bool operator < ( block<T, Sz> const & lhs, block<T, Sz> const & rhs )
{
	for ( std::size_t i = 0; i < Sz; ++i )
	{
		if ( lhs.data_[i] < rhs.data_[i] )
		{
			return true;
		}
		else if ( rhs.data_[i] < lhs.data_[i] )
		{
			return false;
		}
	}

	return false;
}

/**
 * main
 */
int main( void )
{
	typedef block<int, 10> int_array_10;

	int_array_10 a;
	int_array_10 b;

	for ( int_array_10::iterator it = a.begin(); it != a.end(); ++it )
	{
		std::cin >> *it;
	}

	std::copy( a.begin(), a.end(), std::ostream_iterator<int>( std::cout, "\n" ) );
	std::copy( a.begin(), a.end(), b.begin() );
	std::copy( b.begin(), b.end(), std::ostream_iterator<int>( std::cout, "\n" ) );

	return 0;
}
