#pragma once
#include "stdafx.h"
#include "Level.h"

namespace
{
    tbd::ActorFactory;
}
namespace packman
{
    class Maze : public tbd::BaseLevel
    {
    private:
        INT m_size;
        INT m_step;
        INT m_enemies;

        BOOL** m_vals;

    public:
        Maze(INT size, INT enemies, tbd::ActorFactory* factory);

        BOOL VLoad(BOOL block);

        BOOL IsWall(FLOAT, FLOAT);

        INT GetSize(VOID) { return m_size; }

        BOOL VSave(LPCSTR file = NULL);

        FLOAT VGetLoadingProgress(VOID);
        ~Maze(VOID);
    };
}

