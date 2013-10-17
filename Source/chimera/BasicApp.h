#pragma once
#pragma once
#include "stdafx.h"
#include "GameApp.h"

namespace chimera
{
    class BasicApp : public GameApp 
    {
    public:
        CM_DLL_API BasicApp(VOID)
        {

        }

        virtual VOID VCreateLogicAndView(VOID);
    };
}

