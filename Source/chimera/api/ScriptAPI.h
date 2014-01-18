#pragma once
#include "CMTypes.h"

namespace chimera
{
    class IScript
    {
    public:
        virtual bool VInit(void) = 0;

        virtual bool VRunFile(LPCSTR file) = 0;

        virtual bool VRunString(LPCSTR str) = 0;

        virtual ~IScript(void) {}
    };

    class IScriptEventManager
    {
    public:
        //TODO
    };
}
