#include "SoundEngine.h"
#include "Sound.h"
#include "GameApp.h"
#include "Process.h"
#include "Components.h"
#include "Resources.h"
#include "Camera.h"
#include "GameLogic.h"
#include "SceneGraph.h"
#include "math.h"
namespace chimera
{
    SoundEngine::SoundEngine(void)
    {
    }

    void SoundEngine::RegisterSound(std::string material0, std::string material1, std::string soundFile)
    {
        MaterialPair pair;
        pair.m0 = material0;
        pair.m1 = material1;
        m_soundLibrary[pair] = soundFile;
    }

    void SoundEngine::CollisionEventDelegate(event::IEventPtr event)
    {
        std::shared_ptr<event::CollisionEvent> ce = std::static_pointer_cast<event::CollisionEvent>(event);
        ActorId id0 = ce->m_actor0;
        ActorId id1 = ce->m_actor1;
        std::shared_ptr<chimera::Actor> actor0 = chimera::g_pApp->GetLogic()->VFindActor(id0);
        std::shared_ptr<chimera::Actor> actor1 = chimera::g_pApp->GetLogic()->VFindActor(id1);

        if(!actor0 || !actor1)
        {
            return;
        }

        std::shared_ptr<chimera::PhysicComponent> actor0pc = actor0->GetComponent<chimera::PhysicComponent>(chimera::PhysicComponent::COMPONENT_ID).lock();
        std::shared_ptr<chimera::PhysicComponent> actor1pc = actor1->GetComponent<chimera::PhysicComponent>(chimera::PhysicComponent::COMPONENT_ID).lock();

        MaterialPair pair;
        pair.m0 = actor0pc->m_material;
        pair.m1 = actor1pc->m_material;

        auto it = m_soundLibrary.find(pair);
        if(it != m_soundLibrary.end())
        {
            std::string file = it->second;
            chimera::CMResource r(file);
            std::shared_ptr<chimera::ResHandle> handle = chimera::g_pApp->GetCache()->GetHandle(r);

            util::Vec3 position = ce->m_position;
            
            float radius = CLAMP(ce->m_impulse.Length(), 0, 100);
            util::ICamera* camera = chimera::g_pApp->GetHumanView()->GetSceneGraph()->GetCamera().get();
            util::Vec3 pos = camera->GetEyePos() - position;
            int vol = 0;
            float l = pos.Length();

            //if(radius > l)
            {
                vol = (int)(100 * l / radius);
            }

            vol = CLAMP(vol, 0, 100);

            std::shared_ptr<proc::StaticSoundEmitterProcess> proc = std::shared_ptr<proc::StaticSoundEmitterProcess>(new proc::StaticSoundEmitterProcess(position, handle, radius, 0, vol));
            chimera::g_pApp->GetLogic()->AttachProcess(proc);
        }
        else
        {
            //play no sound, or a default one? I guess not...
        }
    }

    void SoundEngine::NewComponentDelegate(event::IEventPtr event)
    {
        std::shared_ptr<event::NewComponentCreatedEvent> pCastEventData = std::static_pointer_cast<event::NewComponentCreatedEvent>(event);
        std::shared_ptr<chimera::Actor> actor = chimera::g_pApp->GetLogic()->VFindActor(pCastEventData->m_actorId);

        if(pCastEventData->m_id == chimera::SoundComponent::COMPONENT_ID)
        {
            std::shared_ptr<chimera::SoundComponent> comp = actor->GetComponent<chimera::SoundComponent>(chimera::SoundComponent::COMPONENT_ID).lock();
            if(!actor->HasComponent<chimera::TransformComponent>(chimera::TransformComponent::COMPONENT_ID))
            {
                LOG_CRITICAL_ERROR("actor needs a transformcomponent");
            }
            std::shared_ptr<chimera::TransformComponent> transform = actor->GetComponent<chimera::TransformComponent>(chimera::TransformComponent::COMPONENT_ID).lock();

            chimera::CMResource r(comp->m_soundFile);
            std::shared_ptr<chimera::ResHandle> handle = chimera::g_pApp->GetCache()->GetHandle(r);
            std::shared_ptr<proc::SoundProcess> proc;
            if(comp->m_emitter)
            {
                proc = std::shared_ptr<proc::SoundEmitterProcess>(new proc::SoundEmitterProcess(actor, transform, handle, comp->m_radius, 0, 0, comp->m_loop));
            }
            else
            {
                proc = std::shared_ptr<proc::SoundProcess>(new proc::SoundProcess(handle, 0, 100, comp->m_loop));
            }
            chimera::g_pApp->GetLogic()->AttachProcess(proc);
            comp->VSetHandled();
        }
    }

    SoundEngine::~SoundEngine(void)
    {
        m_soundLibrary.clear();
    }
}

