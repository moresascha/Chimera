#pragma once
#include <cutil_inline.h>
#include "../Source/chimera/Logger.h"
#include "../Source/chimera/Timer.h"

class cuTimer
{
private:
    chimera::util::HTimer m_timer;
    double m_time;
    long m_ticks;

public:

    cuTimer(void)
    {
        Reset();
    }

    void Reset(void)
    {
        m_time = 0;
        m_ticks = 0;
    }

    void Tick(void)
    {
        cudaDeviceSynchronize();
        m_timer.Start();
    }

    void Tock(void)
    {
        cudaDeviceSynchronize();
        m_timer.Stop();
        m_time += m_timer.GetNanos();
        m_ticks++;
    }

    double GetMillis(void)
    {
        return m_time / 1e6;
    }

    long GetTicks(void)
    {
        assert(m_ticks != 0);
        return m_ticks;
    }

    double GetAverageMillis(void)
    {
        return GetMillis() / GetTicks();
    }

    void Print(const char* info = NULL)
    {
        DEBUG_OUT_A("%s %f\n", info == NULL ? "" : info, GetAverageMillis());
    }
};

class cuScopedTimer : public cuTimer
{
private:
    const char* m_info;
public:
    cuScopedTimer(const char* info = NULL) : m_info(info)
    {
        Tick();
    }

    ~cuScopedTimer(void)
    {
        Tick();
        Print(m_info);
    }
};