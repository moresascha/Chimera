#pragma once

#include "../../Nutty/Nutty/Nutty.h"
#include "../../Nutty/Nutty/DeviceBuffer.h"

#define ADD_TO_ARRAY(name) kernelArg.##name = b.GetDevicePtr()();

template <
    typename T
>
struct SoA 
{
    T kernelArg;

    SoA(void)
    {

    }

    template <
        typename H
    >
    void AddBuffer(LPCSTR name, nutty::DeviceBuffer<H>& b)
    {
        ADD_TO_ARRAY(name)
    }

    T GetKernelArg(void)
    {
        return kernelArg;
    }
};