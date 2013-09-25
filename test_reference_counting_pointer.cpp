#if !defined( REFERENCE_COUNTING_POINTER_HPP )
#define REFERENCE_COUNTING_POINTER_HPP

#include <cassert>

class reference_counted_object
{
public:
	void addref()
	{
		++referenceCount_;
	}

	void release()
	{
		if ( 0 == --referenceCount_ )
		{
			delete this;
		}
	}

protected:
	reference_counted_object() : referenceCount_( 0 ) {}

	/// NOTE: protectedなのでvirtualにする必要なし.
	~reference_counted_object()
	{
		assert( 0 == referenceCount_ )
	}

private:
	std::size_t referenceCount_;
};

template <typename T>
class reference_counting_pointer
{
	typedef T value_type;
	typedef value_type * pointer;
	typedef value_type & reference;

public:
	reference_counting_pointer() : counter_( 0 ) {}

	explicit reference_counting_pointer( value_type * pointee )
			: counter_( new count_holder( pointee ) )
	{
		addref();
	}

	reference_counting_pointer( reference_counting_pointer const & rhs )
			: counter_( rhs.counter_ )
	{
		addref();
	}

	virtual ~reference_counting_pointer()
	{
		release();
	}

public:
	reference_counting_pointer & operator = ( reference_counting_pointer const & rhs )
	{
		if ( this != &rhs )
		{
			release();
			counter_ = rhs.counter_;
			addref();
		}

		return *this;
	}

	pointer operator -> () const { return counter_->pointee_; }
	reference operator * () const { return *counter_->pointee_; }

	bool operator < ( reference_counting_pointer const & rhs ) const { return pointee_ < rhs.pointee_; }
	bool operator == ( reference_counting_pointer const & rhs ) const { return pointee_ == rhs.pointee_; }

private:
	void addref() { if ( counter_ ) { counter_->addref(); } }
	void release() { if ( counter_ ) { counter_->release(); } }

	struct count_holder : public reference_counted_object
	{
		pointer pointee_;

		count_holder( pointer pointee )
				: pointee_( pointee )
		{
		}

		~count_holder()
		{
			delete pointee_;
		}
	};

	count_holder * counter_;
};

#endif // !defined( REFERENCE_COUNTING_POINTER_HPP )
