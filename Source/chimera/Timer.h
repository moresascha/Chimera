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

        class HTimer : public ITimer
        {
        private:
            LARGE_INTEGER m_start;
            LARGE_INTEGER m_lastTick;
            LARGE_INTEGER m_end;
            DOUBLE m_freq;

            double m_timeMillis;
        public:
            HTimer(void) : m_timeMillis(0)
            {
                VReset();
            }

            void Start(void)
            {
                QueryPerformanceCounter(&m_start);
            }

            void Stop(void)
            {
                QueryPerformanceCounter(&m_end);
            }

            double GetTime(double multiplier) const
            {
                return (double)((m_end.QuadPart - m_start.QuadPart) * multiplier) / m_freq;
            }

            double GetNanos(void) const
            {
                return GetTime(1e9);
            }

            double GetMicros(void) const
            {
                return GetTime(1e6);
            }

            double GetMillis(void) const
            {
                return GetTime(1e3);
            }

            double GetSeconds(void) const
            {
                return GetTime(1);
            }

            void VTick(void)
            {
                Stop();
                m_timeMillis = (double)((m_end.QuadPart - m_lastTick.QuadPart) * 1e3) / m_freq;
                QueryPerformanceCounter(&m_lastTick);
            }

            void VReset(void)
            {
                Stop();
                m_timeMillis = 0;
                QueryPerformanceFrequency(&m_start);
                QueryPerformanceCounter(&m_end);
                QueryPerformanceCounter(&m_lastTick);
                m_freq = (double)(m_start.QuadPart);
                Start();
            }

            float VGetFPS(void) const
            {
                return 0;
            }

            ulong VGetTime(void) const
            {
                return (ulong)(GetMillis() + 0.5);
            }

            ulong VGetLastMillis(void) const
            {
                return (ulong)(m_timeMillis + 0.5);
            }

            ulong VGetLastMicros(void) const
            {
                return (ulong)(1000 * VGetLastMillis() + 0.5); //todo
            }
        };
    }
}