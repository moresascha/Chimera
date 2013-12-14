#pragma once
#include "../../Nutty/Nutty/Nutty.h"
#include "../../Nutty/Nutty/DeviceBuffer.h"

template <
    typename T
>
class DoubleBuffer
{
private:
    nutty::DeviceBuffer<T> m_buffer[2];
    byte m_current;

public:
    DoubleBuffer(void) :  m_current(0)
    {

    }

    void Resize(size_t size)
    {
        for(byte i = 0; i < 2; ++i)
        {
            m_buffer[i].Resize(size);
        }
    }

    nutty::DeviceBuffer<T>& Get(byte index)
    {
        return m_buffer[index];
    }

    nutty::DeviceBuffer<T>& GetCurrent(void)
    {
        return m_buffer[m_current];
    }

    void Toggle(void)
    {
        m_current = (m_current + 1) % 2;
    }
};
