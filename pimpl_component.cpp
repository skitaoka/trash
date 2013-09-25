// header ----------------------------------------------------------------------
struct ComponentImpl;

class Component
{
public:
	static Component& getInstance()
	{
		assert( instance );
		return *instance;
	}

	static void initialize()
	{
		assert( !instance );
		instance = new Component();
	}

	static void cleanup()
	{
		assert( instance );
		delete instance;
		instance = 0;
	}

	void run();

private:
	Component();
	~Component();

	ComponentImpl* _pimpl;

	static Component* instance;
};

// source ----------------------------------------------------------------------
struct ComponentImpl
{
	inline void run()
	{
		puts( "Hello world!" );
	}
};

Component* Component::instance = 0;

Component::Component()
{
	_pimpl = new ComponentImpl();
}

~Component()
{
	delete _pimpl;
}


void Component::run()
{
	_pimpl->run();
}
