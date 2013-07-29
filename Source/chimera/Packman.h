#pragma once
#include "stdafx.h"
#include "GameApp.h"
#include "Actor.h"
#include "GameLogic.h"
#include "Components.h"
#include "GameApp.h"
#include "GuiHelper.h"
#include "Input.h"
#include "Commands.h"
#include "d3d.h"
#include "Event.h"

namespace packman
{
    class Packman : public app::GameApp
    {
        VOID VCreateLogicAndView(VOID);
    };

    class AIComponent : public tbd::ActorComponent
    {
    public:
        CONST static ComponentId COMPONENT_ID;
        LPCSTR VGetName(VOID) { return "AIComponent"; }
        virtual ComponentId GetComponentId(VOID) CONST { return COMPONENT_ID; };
    };
}

