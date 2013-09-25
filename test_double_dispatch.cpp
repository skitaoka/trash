/**
 * gcc 
 */

#include <iostream>
#include <map>
#include <stdexcept>

/**
 * ゲームオブジェクト
 */
class GameObject
{
public:
	virtual ~GameObject() {}
};

/**
 * 未定義の衝突エラー
 */
class UnknownCollisionException : public std::runtime_error
{
public:
	UnknownCollisionException( GameObject const & obj1, GameObject const & obj2 )
			: std::runtime_error( "UnknownCollisionException" )
			, obj1_( typeid( obj1 ).name() )
			, obj2_( typeid( obj2 ).name() )
	{
	}

	virtual ~UnknownCollisionException() throw() {}

private:
	std::string const obj1_;
	std::string const obj2_;
};

/**
 * 衝突関数を得るマップ
 */
class CollisionMap
{
public:
	typedef void (*CollisionFunctionPtr)( GameObject&, GameObject& );

private:
	typedef std::pair<std::string, std::string> string_pair;
	typedef std::map<string_pair, CollisionFunctionPtr> hit_map;

public:
	void operator () ( GameObject & obj1, GameObject & obj2 )
	{
		CollisionFunctionPtr pHitFunction = lookUp( typeid( obj1 ).name(), typeid( obj2 ).name() );

		if ( pHitFunction )
		{
			pHitFunction( obj1, obj2 );
		}
		else
		{
			throw UnknownCollisionException( obj1, obj2 );
		}
	}

	void addEntry( std::string const & klass1, std::string const & klass2,
			CollisionFunctionPtr collisionFunction, bool isSymmetric = true )
	{
		hitMap_[ string_pair( klass1, klass2 ) ] = collisionFunction;
		if ( isSymmetric )
		{
			hitMap_[ string_pair( klass2, klass1 ) ] = collisionFunction;
		}
	}

	void removeEntry( std::string const & klass1, std::string const & klass2 )
	{
		hitMap_.erase( string_pair( klass1, klass2 ) );
	}

	static CollisionMap & getInstance()
	{
		static CollisionMap collisionMap;
		return collisionMap;
	}

private:
	CollisionFunctionPtr lookUp( std::string const & klass1, std::string const & klass2 )
	{
		hit_map::iterator it = hitMap_.find( string_pair( klass1, klass2 ) );

		if ( it == hitMap_.end() )
		{
			return	0;
		}

		return (*it).second;
	}

private:
	CollisionMap() {}
	CollisionMap( const CollisionMap& );
	CollisionMap & operator = ( const CollisionMap& );

private:
	hit_map hitMap_;
};

/**
 * 安全に衝突関数を登録するレジスタ
 */
struct RegistorCollisionFunction
{
	RegistorCollisionFunction( std::string const & klass1, std::string const & klass2,
			CollisionMap::CollisionFunctionPtr collisionFunction, bool isSymmetric = true )
	{
		CollisionMap::getInstance().addEntry( klass1, klass2, collisionFunction, isSymmetric );
	}
};

//
// 実装
//
class Foo : public GameObject {};
class Bar : public GameObject {};
class Baz : public GameObject {};

void FooBar( GameObject &, GameObject & ) { std::cout << "FooBar" << std::endl; }
void FooBaz( GameObject &, GameObject & ) { std::cout << "FooBaz" << std::endl; }
void BarBaz( GameObject &, GameObject & ) { std::cout << "BarBaz" << std::endl; }

RegistorCollisionFunction cf1( "3Foo", "3Bar", &FooBar );
RegistorCollisionFunction cf2( "3Foo", "3Baz", &FooBaz );
RegistorCollisionFunction cf3( "3Bar", "3Baz", &BarBaz );

int main( void )
{
	std::auto_ptr<GameObject> obj1( new Foo() );
	std::auto_ptr<GameObject> obj2( new Baz() );
	std::auto_ptr<GameObject> obj3( new Bar() );

	try
	{
		CollisionMap::getInstance()( *obj1, *obj2 );
		CollisionMap::getInstance()( *obj2, *obj1 );
		CollisionMap::getInstance()( *obj2, *obj3 );
		CollisionMap::getInstance()( *obj3, *obj2 );
		CollisionMap::getInstance()( *obj3, *obj1 );
		CollisionMap::getInstance()( *obj1, *obj3 );
	}
	catch ( UnknownCollisionException & e )
	{
		std::cout << e.what() << std::endl;
		return 1;
	}

	return 0;
}
