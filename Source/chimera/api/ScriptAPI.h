#pragma once
#include "CMTypes.h"

namespace chimera
{
    class IScript
    {
    public:
        virtual BOOL VInit(VOID) = 0;

        virtual VOID VRunFile(LPCSTR file) = 0;

        virtual VOID VRunString(LPCSTR str) = 0;

        virtual ~IScript(VOID) {}
    };

    class IScriptEventManager
    {
    public:
        //TODO
    };
}