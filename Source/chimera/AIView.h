#pragma once
#include "Actor.h"
#include "Maze.h"
#include "Timer.h"
#include "GameView.h"

namespace tbd
{
    class TransformComponent;
}

namespace packman
{
    class EnemyAIView : public tbd::IGameView
    {
    private:
        packman::Maze* m_pLevel;
        tbd::TransformComponent* m_pTransform;
        util::Timer m_timer;
        util::Vec3 m_dir;
    public:
        EnemyAIView(packman::Maze* level);

        tbd::GameViewType VGetType(VOID) CONST;
        VOID VOnAttach(UINT viewId, std::shared_ptr<tbd::Actor> actor);
        VOID VOnUpdate(ULONG deltaMillis);
        VOID PickDir(VOID);
    };
}

