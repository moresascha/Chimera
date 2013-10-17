#include "Timer.h"
#include <math.h>
#include <sstream>

#pragma comment (lib, "winmm.lib")

namespace chimera
{
	namespace util 
	{
		Timer::Timer(VOID) 
		{
			this->VReset();
		}

		VOID Timer::VTick(VOID) 
		{
			m_lastCounter = m_currentCounter;
			m_currentCounter = timeGetTime();
			//QueryPerformanceCounter((LARGE_INTEGER*)&m_currentCounter);

			m_framesCount++;
			if((m_currentCounter - m_lastFramesStart) >= 1000)
			{
				m_lastFramesStart = m_currentCounter;
				m_lastFramesCount = m_framesCount;
				m_framesCount = 0;
			}
		}

		ULONG Timer::VGetTime(VOID) CONST
		{
			return (ULONG)(m_currentCounter - m_start);// / m_frequ);
		}

		/*
		VOID Timer::Tock(VOID)
		{
			this->m_lastTock = clock();
			this->m_accuTickTock += (this->m_lastTock - this->m_lastTick);
		}

		VOID Timer::TickTock(VOID)
		{
			if(m_tickTock)
			{
				Tick();
			}
			else
			{
				Tock();
				this->m_ticks++;
			}
			this->m_tickTock = !this->m_tickTock;
		}
		 */
		VOID Timer::VReset(VOID) 
		{
			m_framesCount = 0;
			m_start = timeGetTime();
			/*QueryPerformanceFrequency((LARGE_INTEGER*)&m_frequ);
			QueryPerformanceCounter((LARGE_INTEGER*)&m_start); */
			m_lastFramesStart = m_currentCounter = m_lastCounter = m_start;
		}

		FLOAT Timer::VGetFPS(VOID) CONST
		{
			return (FLOAT)m_lastFramesCount;
		}

		ULONG Timer::VGetLastMillis() CONST
		{
			return (m_currentCounter - m_lastCounter);//(ULONG)((1e3 * (m_currentCounter - m_lastCounter)) / m_frequ);
		}

		ULONG Timer::VGetLastMicros() CONST
		{
			return (ULONG)(1e3 * (m_currentCounter - m_lastCounter));// / m_frequ);
		}

		Timer::~Timer(VOID) 
		{

		}
	}
}