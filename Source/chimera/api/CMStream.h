#pragma once
#include "CMTypes.h"

namespace chimera
{
    class ICMStream
    {
    public:
        virtual void Close(void) = 0;
    };
}