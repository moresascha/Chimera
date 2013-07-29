#pragma once
#pragma once
#include "stdafx.h"
#include "Actor.h"
#include "GameLogic.h"
#include "Components.h"
#include "GameApp.h"
#include "GuiHelper.h"
#include "Input.h"
#include "Commands.h"
#include "d3d.h"

namespace app
{
    class BasicApp : public app::GameApp 
    {
        virtual VOID VCreateLogicAndView(VOID);
    };
}

