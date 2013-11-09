#pragma once
#include "CMTypes.h"

namespace chimera
{
    class IScript
    {
    public:
        virtual BOOL VInit(VOID) = 0;

        virtual BOOL VRunFile(LPCSTR file) = 0;

        virtual BOOL VRunString(LPCSTR str) = 0;

        virtual ~IScript(VOID) {}
    };

    class IScriptEventManager
    {
    public:
        //TODO
    };
}
