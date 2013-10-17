#pragma once
#include "stdafx.h"
#include "GameLogic.h"
#include "Event.h"

namespace packman
{
    class Maze;
    class PackmanLogic : public chimera::BaseGameLogic
    {
    public:

        Maze* GetMaze(VOID)
        {
            return (Maze*)m_pLevel;
        }

        BOOL VInit(VOID);

        VOID ComponentCreatedDelegate(chimera::IEventPtr data);

        ~PackmanLogic(VOID);
    };
}

