#include "AIView.h"
#include "Level.h"
#include "Components.h"
#include "GameApp.h"
#include "GameLogic.h"
#include "ActorFactory.h"
#include "Event.h"
#include "EventManager.h"
#include "LuaHelper.h"
#include "GameApp.h"

namespace packman
{
    EnemyAIView::EnemyAIView(packman::Maze* level) : m_pLevel(level), m_dir(0,0,1)
    {

    }

    VOID EnemyAIView::VOnAttach(UINT viewId, std::shared_ptr<tbd::Actor> actor)
    {
        tbd::IGameView::VOnAttach(viewId, actor);
        m_pTransform = m_actor->GetComponent<tbd::TransformComponent>(tbd::TransformComponent::COMPONENT_ID).lock().get();
    }

    VOID EnemyAIView::PickDir(VOID)
    {
        FLOAT dx = 0;
        FLOAT dz = 0;

        INT size = m_pLevel->GetSize();
        FLOAT radius = 0.5f;
        dx = (rand() / (FLOAT) RAND_MAX) < 0.5f ? -1.0f : 1.0f;
        dx *=  ((rand() / (FLOAT) RAND_MAX) < 0.5f ? radius : 0.0f);

        if(dx == 0)
        {
            dz = (rand() / (FLOAT) RAND_MAX) < 0.5f ? -1.0f : 1.0f;
            dz *=  radius;
        }

        m_dir = util::Vec3 (dx, 0, dz);
    }

    tbd::GameViewType EnemyAIView::VGetType(VOID) CONST
    {
        return tbd::AI;
    }

    VOID ScriptOnUpdate(ActorId id, CONST util::Vec3& position, INT levelSize)
    {
        LuaPlus::LuaObject o = app::g_pApp->GetScript()->GetState()->GetGlobal("OnUpdate");
        if(!o.IsNil() && o.IsFunction())
        {
            LuaPlus::LuaFunction<VOID> function = app::g_pApp->GetScript()->GetState()->GetGlobal("OnUpdate");

            function(tbd::script::GetIntegerObject(id), tbd::script::GetVec3Object(position), tbd::script::GetIntegerObject(levelSize));
        }
    }

    VOID EnemyAIView::VOnUpdate(ULONG deltaMillis)
    {
        if(m_timer.GetTime() > 16)
        {
            /*util::Vec3 newPos = m_pTransform->GetTransformation()->GetTranslation() + m_dir;
            if(!m_pLevel->IsWall(newPos.x + (m_dir.x > 0 ? +1.5f : 0), newPos.z + (m_dir.z > 0 ? +1.5f : 0)))
            {
                event::IEventPtr moveEvent = std::shared_ptr<event::MoveActorEvent>(new event::MoveActorEvent(m_actor->GetId(), util::Vec3(m_dir.x, 0, m_dir.z), util::Vec3(), TRUE));
                event::IEventManager::Get()->VQueueEvent(moveEvent);
            }
            else
            {
                PickDir();
            } */
            ScriptOnUpdate(m_actor->GetId(), m_pTransform->GetTransformation()->GetTranslation(), m_pLevel->GetSize());
            m_timer.Reset();
        }
        m_timer.Tick();
    }
}