#define guard(func)											\
	{														\
		static const TCHAR __FUNC_NAME__[] = TEXT(#func);	\
		try {

#define unguard												\
		} catch (...) {										\
			appUnwindf( TEXT("%s"), __FUNC_NAME__ );		\
			throw;											\
		}													\
	}

void appUnwindf( char*, char* )
{
	// グローバルなスタックに自分の関数名を順に追加していく
}

UBOOL Step()
{
	guard( Foo::Step )
	{
		// ...
	}
	unguard;
}
