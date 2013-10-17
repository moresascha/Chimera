#pragma once
#include "Actor.h"
#include "Maze.h"
#include "Timer.h"
#include "GameView.h"

namespace chimera
{
    class TransformComponent;
}

namespace packman
{
    class EnemyAIView : public chimera::IGameView
    {
    private:
        packman::Maze* m_pLevel;
        chimera::TransformComponent* m_pTransform;
        chimera::util::Timer m_timer;
        chimera::util::Vec3 m_dir;
    public:
        EnemyAIView(packman::Maze* level);

        chimera::GameViewType VGetType(VOID) CONST;
        VOID VOnAttach(UINT viewId, std::shared_ptr<chimera::Actor> actor);
        VOID VOnUpdate(ULONG deltaMillis);
        VOID PickDir(VOID);
    };
}

