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

// ���s����Ɣ� 0 �̒l��Ԃ�.
inline int pthread_create(
	pthread_t          * const thread,
	pthread_attr_t     * const attr,
	unsigned (_stdcall * const start_routine)(void*),
	void               * const arg) 
{
	unsigned thrdaddr; // �X���b�h���ʎq (HANDLE ���擾����̂ŕs�v)
	*thread = reinterpret_cast<HANDLE>(::_beginthreadex(NULL, 0, start_routine, arg, 0, &thrdaddr));
	return *thread == NULL;
}

// ���s����Ɣ� 0 �̒l��Ԃ�.
inline int pthread_join(
	pthread_t const thread,
	void   ** const thread_return)
{
	return ::WaitForSingleObject(thread, INFINITE) == WAIT_FAILED;
}

// ��� 0 ��Ԃ�.
inline int pthread_mutex_init(
	pthread_mutex_t     * const mutex,
	pthread_mutexattr_t * const mutexattr)
{
	::InitializeCriticalSection(mutex);
	return 0;
}

// ���s����Ɣ� 0 �̒l��Ԃ� ... �d�l�����{�����ł͏�� 0 ��Ԃ�.
inline int pthread_mutex_destroy(pthread_mutex_t * const mutex)
{
	::DeleteCriticalSection(mutex);
	return 0;
}

// ���s����Ɣ� 0 �̒l��Ԃ� ... �d�l�����{�����ł͏�� 0 ��Ԃ�.
inline int pthread_mutex_lock(pthread_mutex_t * const mutex)
{
	::EnterCriticalSection(mutex);
	return 0;
}

// ���s����Ɣ� 0 �̒l��Ԃ� ... �d�l�����{�����ł͏�� 0 ��Ԃ�.
inline int pthread_mutex_unlock(pthread_mutex_t * const mutex)
{
	::LeaveCriticalSection(mutex);
	return 0;
}

// ��� 0 ��Ԃ�.
inline int pthread_cond_init(
	pthread_cond_t     * const cond,
	pthread_condattr_t * const cond_attr)
{
	// �������Z�b�g, ��V�O�i����Ԃŏ�����
	*cond = ::CreateEvent(NULL, FALSE, FALSE, NULL);
	return 0;
}

// ���s����Ɣ� 0 �̒l��Ԃ�.
inline int pthread_cond_destroy(pthread_cond_t const * const cond)
{
	return ::CloseHandle(*cond);
}

// ��� 0 ��Ԃ�.
inline int pthread_cond_signal(pthread_cond_t const * const cond)
{
	::SetEvent(*cond);
	return 0;
}

// ���s����Ɣ� 0 �̒l��Ԃ�.
inline int pthread_cond_wait(
	pthread_cond_t  * const cond,
	pthread_mutex_t * const mutex)
{
	pthread_mutex_unlock(mutex);
	DWORD const retval = ::WaitForSingleObject(*cond, INFINITE);
	pthread_mutex_lock(mutex);
	return retval == WAIT_FAILED;
}

// ���s����Ɣ� 0 �̒l��Ԃ�.
inline int pthread_cond_timedwait(
	pthread_cond_t        * const cond,
	pthread_mutex_t       * const mutex,
	struct timespec const * const abstime)
{
	return pthread_cond_wait(cond, mutex);
}

#endif//AKA_CUDAMINER_PTHREAD_INCLUDED
