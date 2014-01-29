#pragma once

#ifndef AKA_CUDAMINER_PTHREAD_INCLUDED
#define AKA_CUDAMINER_PTHREAD_INCLUDED

#include <Windows.h>
#include <process.h>

typedef HANDLE           pthread_t;           //
typedef int              pthread_attr_t;      // dummy
typedef CRITICAL_SECTION pthread_mutex_t;     // 
typedef int              pthread_mutexattr_t; // dummy
typedef HANDLE           pthread_cond_t;      //
typedef int              pthread_condattr_t;  // dummy

// 失敗すると非 0 の値を返す.
inline int pthread_create(
	pthread_t          * const thread,
	pthread_attr_t     * const attr,
	unsigned (_stdcall * const start_routine)(void*),
	void               * const arg) 
{
	unsigned thrdaddr; // スレッド識別子 (HANDLE を取得するので不要)
	*thread = reinterpret_cast<HANDLE>(::_beginthreadex(NULL, 0, start_routine, arg, 0, &thrdaddr));
	return *thread == NULL;
}

// 失敗すると非 0 の値を返す.
inline int pthread_join(
	pthread_t const thread,
	void   ** const thread_return)
{
	return ::WaitForSingleObject(thread, INFINITE) == WAIT_FAILED;
}

// 常に 0 を返す.
inline int pthread_mutex_init(
	pthread_mutex_t     * const mutex,
	pthread_mutexattr_t * const mutexattr)
{
	::InitializeCriticalSection(mutex);
	return 0;
}

// 失敗すると非 0 の値を返す ... 仕様だが本実装では常に 0 を返す.
inline int pthread_mutex_destroy(pthread_mutex_t * const mutex)
{
	::DeleteCriticalSection(mutex);
	return 0;
}

// 失敗すると非 0 の値を返す ... 仕様だが本実装では常に 0 を返す.
inline int pthread_mutex_lock(pthread_mutex_t * const mutex)
{
	::EnterCriticalSection(mutex);
	return 0;
}

// 失敗すると非 0 の値を返す ... 仕様だが本実装では常に 0 を返す.
inline int pthread_mutex_unlock(pthread_mutex_t * const mutex)
{
	::LeaveCriticalSection(mutex);
	return 0;
}

// 常に 0 を返す.
inline int pthread_cond_init(
	pthread_cond_t     * const cond,
	pthread_condattr_t * const cond_attr)
{
	// 自動リセット, 非シグナル状態で初期化
	*cond = ::CreateEvent(NULL, FALSE, FALSE, NULL);
	return 0;
}

// 失敗すると非 0 の値を返す.
inline int pthread_cond_destroy(pthread_cond_t const * const cond)
{
	return ::CloseHandle(*cond);
}

// 常に 0 を返す.
inline int pthread_cond_signal(pthread_cond_t const * const cond)
{
	::SetEvent(*cond);
	return 0;
}

// 失敗すると非 0 の値を返す.
inline int pthread_cond_wait(
	pthread_cond_t  * const cond,
	pthread_mutex_t * const mutex)
{
	pthread_mutex_unlock(mutex);
	DWORD const retval = ::WaitForSingleObject(*cond, INFINITE);
	pthread_mutex_lock(mutex);
	return retval == WAIT_FAILED;
}

// 失敗すると非 0 の値を返す.
inline int pthread_cond_timedwait(
	pthread_cond_t        * const cond,
	pthread_mutex_t       * const mutex,
	struct timespec const * const abstime)
{
	return pthread_cond_wait(cond, mutex);
}

#endif//AKA_CUDAMINER_PTHREAD_INCLUDED
