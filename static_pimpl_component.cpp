// header ----------------------------------------------------------------------
class Component
{
public:
	static void initialize();
	static void cleanup();
	static void run();
};

// source ----------------------------------------------------------------------
namespace {

struct ComponentImpl
{
	inline void run()
	{
		puts( "Hello world!" );
	}
}

ComponentImp* component_pimpl = 0;

} // local namespace

void Component::initialize()
{
	assert( !component_pimpl );
	component_pimpl = new ComponentImp();
}

void Component::cleanup()
{
	assert( component_pimpl );
	delete component_pimpl;
	component_pimpl = 0;
}

void Component::run()
{
	component_pimpl->run();
}
