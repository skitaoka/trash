#if defined( WIN32 )
#	if defined( __cplusplus )
#		define DLLEXPORT extern "C" __declspec( dllexport )
#	else
#		define DLLEXPORT __declspec( dllexport )
#	endif // defined( __cplusplus )
#else
#	define DLLEXPORT
#endif // defined( WIN32 )

#if defined( __APPLE__ ) && defined( __MACH__ )
#	include <mach-o/dyld.h>
#elif defined( WIN32 )
#	include <windows.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct dl_module_t
{
#if defined( __APPLE__ ) && defined( __MACH__ )

	NSModule	module;

#elif defined( WIN32 )

	HINSTANCE	module;

#elif defined( LINUX )

	void*		module;

#else

	int			module;

#endif
};

extern bool  dl_load( dl_module_t* module, const char* filename );
extern void* dl_func( dl_module_t* module, const char* filename );

#ifdef __cplusplus
}
#endif

bool dl_load( dl_module_t* module, const char* filename )
{
#if defined( __APPLE__ ) && defined( __MACH__ )

		NSObjectFileImage	objFile;
		NSSymbol			sym;

		if ( NSCreateObjectFileImageFromFile( filename, &objFile ) == NSObjectFileImageSuccess ) {
			return false;
		}

		module->module = NSLinkModule( objFile, filename, NSLINKMODULE_OPTION_BINDNOW );
		if ( !module->module ) {
			return false;
		}

#elif defined( WIN32 )

		module->module = LoadLibrary( filename );
		if ( !module->module ) {
			return false;
		}

#elif defined( LINUX )

		module->module = dlopen( filename, RTLD_LAZY );
		if ( !module->module ) {
			return false;
		}

#endif

	return true;
}


void* dl_func( dl_module_t* module, const char* filename )
{
	char symbol[1024];

#if defined( __APPLE__ ) && defined( __MACH__ )

		sprintf( symbol, "_%s", funcname );

		NSSymbol sym = NSLookupSymbolInModule( module->module, symbol );
		if ( !sym ) {
			return 0;
		}

		return NSAddressOfSymbol( sym );

#elif defined( WIN32 )

		return GetProcAddress( module->module, funcname );

#elif defined( LINUX )

		void* func = dlsym( module->module, funcname );

		if ( dlerror() ) {
			return 0;
		}

		return func;
#endif
}
