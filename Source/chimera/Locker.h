#pragma once
#include "stdafx.h"
namespace util {
class Locker {
private:
    CRITICAL_SECTION m_criticalSection;
public:
    Locker(VOID) {
        InitializeCriticalSection(&m_criticalSection);
    }

    VOID Lock(VOID) {
        EnterCriticalSection(&m_criticalSection);
    }

    VOID Unlock(VOID) {
        LeaveCriticalSection(&m_criticalSection);
    }

    ~Locker(VOID) {
        DeleteCriticalSection(&m_criticalSection);
    }
};
};

