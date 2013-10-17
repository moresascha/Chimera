#pragma once
#include "stdafx.h"
#include <ctime>
namespace chimera
{
    namespace util 
    {
        class Timer : public ITimer
        {
        private:
            ULONG m_framesCount;
            ULONG m_lastFramesCount;
            UINT m_start;
            UINT m_lastFramesStart;
            UINT m_lastCounter;
            UINT m_currentCounter;

        public:
            Timer(VOID);
    
            VOID VTick(VOID);

            VOID VReset(VOID);

            FLOAT VGetFPS(VOID) CONST;

            ULONG VGetTime(VOID) CONST;

            ULONG VGetLastMillis(VOID) CONST;

            ULONG VGetLastMicros(VOID) CONST;
         
            ~Timer(VOID);
        };

        class HTimer //todo: public ITimer
        {
        private:
            LARGE_INTEGER m_start;
            LARGE_INTEGER m_end;
            DOUBLE m_freq;
        public:
            HTimer(VOID)
            {
                QueryPerformanceFrequency(&m_start);
                m_freq = (DOUBLE)(m_start.QuadPart);
            }

            VOID Start(VOID)
            {
                QueryPerformanceCounter(&m_start);
            }

            VOID Stop(VOID)
            {
                QueryPerformanceCounter(&m_end);
            }

            DOUBLE GetTime(DOUBLE multiplier)
            {
                return (DOUBLE)((m_end.QuadPart - m_start.QuadPart) * multiplier) / m_freq;
            }

            DOUBLE GetNanos(VOID)
            {
                return GetTime(1e9);
            }

            DOUBLE GetMicros(VOID)
            {
                return GetTime(1e6);
            }

            DOUBLE GetMillis(VOID)
            {
                return GetTime(1e3);
            }

            DOUBLE GetSeconds(VOID)
            {
                return GetTime(1);
            }
        };
    }
}