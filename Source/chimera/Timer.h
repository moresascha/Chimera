#pragma once
#include "stdafx.h"
#include <ctime>
namespace util {
class DLL_EXPORT Timer
{
private:
    ULONG m_framesCount;
    ULONG m_lastFramesCount;
    /*
    LONGLONG m_lastFramesStart;
    LONGLONG m_start;
    LONGLONG m_frequ;
    LONGLONG m_currentCounter;
    LONGLONG m_lastCounter; */

    UINT m_start;
    UINT m_lastFramesStart;
    UINT m_lastCounter;
    UINT m_currentCounter;

public:
    Timer(VOID);
    
    VOID Tick(VOID);
    /*
    VOID Tock(VOID);
    
    VOID TickTock(VOID);
    */
    VOID Reset(VOID);

    FLOAT GetFPS(VOID) CONST;

    ULONG GetTime(VOID) CONST;

    ULONG GetLastMillis(VOID) CONST;

    ULONG GetLastMicros(VOID) CONST;
         
    ~Timer(VOID);
};

}

