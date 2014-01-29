/*
 * Null Object Sample
 */

namespace detail
{
	class __null_object
	{
	public:
		inline __null_object() {}
		template <typename T> operator T * () const { return 0; }
		template <typename C, typename T> operator T C:: * () const { return 0; }

	private:
		void operator & () const;
	};
}

detail::__null_object const null;

int main( void )
{
	int * i = new int( 2 );

	if ( i )
	{
		delete i;
		i = null;
	}

	double * d = new double( 2 );

	if ( d )
	{
		delete d;
		d = null;
	}

	return 0;
}
