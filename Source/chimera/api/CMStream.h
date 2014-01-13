#pragma once
#include "CMTypes.h"

namespace chimera
{
    class ICMStream
    {
    public:
        virtual VOID Close(VOID) = 0;
    };
}