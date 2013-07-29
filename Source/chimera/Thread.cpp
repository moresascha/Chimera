#include "Thread.h"

namespace util {

Thread::Thread(LPVOID param, DWORD (*threadFunc)(LPVOID param)) : m_param(param), m_fpThreadFunc(threadFunc) {

}

DWORD WINAPI Thread::Run(Thread* thread) {
    DWORD result = 0;
    if(thread->m_fpThreadFunc)
    {
        result = thread->m_fpThreadFunc(thread->m_param);
    }
    thread->OnDone();
    return result;
}

VOID Thread::Start(VOID) {
    CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)(this->Run), this, 0, NULL);
}

VOID Thread::OnDone(VOID) {
    if(this->m_pfCallBack)
    {
        this->m_pfCallBack(this->m_param);
    }
}

Thread::~Thread(VOID) {
}

}
