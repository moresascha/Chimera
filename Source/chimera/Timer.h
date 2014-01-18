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
            ulong m_framesCount;
            ulong m_lastFramesCount;
            uint m_start;
            uint m_lastFramesStart;
            uint m_lastCounter;
            uint m_currentCounter;

        public:
            Timer(void);
    
            void VTick(void);

            void VReset(void);

            float VGetFPS(void) const;

            ulong VGetTime(void) const;

            ulong VGetLastMillis(void) const;

            ulong VGetLastMicros(void) const;
         
            ~Timer(void);
        };

        class HTimer //todo: public ITimer
        {
        private:
            LARGE_INTEGER m_start;
            LARGE_INTEGER m_end;
            DOUBLE m_freq;
        public:
            HTimer(void)
            {
                QueryPerformanceFrequency(&m_start);
                m_freq = (DOUBLE)(m_start.QuadPart);
            }

            void Start(void)
            {
                QueryPerformanceCounter(&m_start);
            }

            void Stop(void)
            {
                QueryPerformanceCounter(&m_end);
            }

            DOUBLE GetTime(DOUBLE multiplier)
            {
                return (DOUBLE)((m_end.QuadPart - m_start.QuadPart) * multiplier) / m_freq;
            }

            DOUBLE GetNanos(void)
            {
                return GetTime(1e9);
            }

            DOUBLE GetMicros(void)
            {
                return GetTime(1e6);
            }

            DOUBLE GetMillis(void)
            {
                return GetTime(1e3);
            }

            DOUBLE GetSeconds(void)
            {
                return GetTime(1);
            }
        };
    }
}