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

    void EnemyAIView::VOnAttach(uint viewId, std::shared_ptr<chimera::Actor> actor)
    {
        chimera::IGameView::VOnAttach(viewId, actor);
        m_pTransform = m_actor->GetComponent<chimera::TransformComponent>(chimera::TransformComponent::COMPONENT_ID).lock().get();
    }

    void EnemyAIView::PickDir(void)
    {
        float dx = 0;
        float dz = 0;

        int size = m_pLevel->GetSize();
        float radius = 0.5f;
        dx = (rand() / (float) RAND_MAX) < 0.5f ? -1.0f : 1.0f;
        dx *=  ((rand() / (float) RAND_MAX) < 0.5f ? radius : 0.0f);

        if(dx == 0)
        {
            dz = (rand() / (float) RAND_MAX) < 0.5f ? -1.0f : 1.0f;
            dz *=  radius;
        }

        m_dir = util::Vec3 (dx, 0, dz);
    }

    chimera::ViewType EnemyAIView::VGetType(void) const
    {
        return chimera::eProjectionType_AI;
    }

    void ScriptOnUpdate(ActorId id, const util::Vec3& position, int levelSize)
    {
        LuaPlus::LuaObject o = chimera::g_pApp->GetScript()->GetState()->GetGlobal("OnUpdate");
        if(!o.IsNil() && o.IsFunction())
        {
            LuaPlus::LuaFunction<void> function = chimera::g_pApp->GetScript()->GetState()->GetGlobal("OnUpdate");

            function(chimera::script::GetIntegerObject(id), chimera::script::GetVec3Object(position), chimera::script::GetIntegerObject(levelSize));
        }
    }

    void EnemyAIView::VOnUpdate(ulong deltaMillis)
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