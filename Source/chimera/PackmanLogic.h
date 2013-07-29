#pragma once
#include "stdafx.h"
#include "GameLogic.h"
#include "Event.h"

namespace packman
{
    class Maze;
    class PackmanLogic : public tbd::BaseGameLogic
    {
    public:

        Maze* GetMaze(VOID)
        {
            return (Maze*)m_pLevel;
        }

        BOOL VInit(VOID);

        VOID ComponentCreatedDelegate(event::IEventPtr data);

        ~PackmanLogic(VOID);
    };
}

