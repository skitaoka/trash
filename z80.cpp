void cpu8086()
{
#define CALL( label ) context_statck.push_back(__LINE__) ; goto label ; case __LINE__ :
#define RET() break

	union 
	{
		short x;
		struct
		{
			char l;
			char h;
		};
	};

	register short ax;
	register short bx;
	register short cx;
	register short dx;

	register short ds;
	register short es;
	register short ss;
	register short cs;

	register short sp;
	register short bp;
	register short ip;

	register short si;
	register short di;

	std::vector<std::size_t> context_stack;

	context_statck.push_back( -1 )
	for ( ; ; ) {
		int context = context_statck.back();
		context_stack().pop_back();
		switch (context) {
		// プログラムエントリーポイント
		case -1:
			CALL(routine_1);
			CALL(routine_2);
			CALL(routine_1);
			RET();

		routine_1:
			CALL(routine_2);
			CALL(routine_2);
			RET();

		routine_2:
			RET();
		}

		if ( context_stack.empty() ) break; // 外側のfor を break
	}

#undef CALL
#undef RET
}
