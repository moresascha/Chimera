#include "SoundEngine.h"
#include "Sound.h"
#include "GameApp.h"
#include "Process.h"
#include "Components.h"
#include "Resources.h"
#include "Camera.h"
#include "GameLogic.h"
#include "SceneGraph.h"
namespace tbd
{
    SoundEngine::SoundEngine(VOID)
    {
    }

    VOID SoundEngine::ResgisterSound(std::string material0, std::string material1, std::string soundFile)
    {
        MaterialPair pair;
        pair.m0 = material0;
        pair.m1 = material1;
        m_soundLibrary[pair] = soundFile;
    }

    VOID SoundEngine::CollisionEventDelegate(event::IEventPtr event)
    {
        std::shared_ptr<event::CollisionEvent> ce = std::static_pointer_cast<event::CollisionEvent>(event);
        ActorId id0 = ce->m_actor0;
        ActorId id1 = ce->m_actor1;
        std::shared_ptr<tbd::Actor> actor0 = app::g_pApp->GetLogic()->VFindActor(id0);
        std::shared_ptr<tbd::Actor> actor1 = app::g_pApp->GetLogic()->VFindActor(id1);

        if(!actor0 || !actor1)
        {
            return;
        }

        std::shared_ptr<tbd::PhysicComponent> actor0pc = actor0->GetComponent<tbd::PhysicComponent>(tbd::PhysicComponent::COMPONENT_ID).lock();
        std::shared_ptr<tbd::PhysicComponent> actor1pc = actor1->GetComponent<tbd::PhysicComponent>(tbd::PhysicComponent::COMPONENT_ID).lock();

        MaterialPair pair;
        pair.m0 = actor0pc->m_material;
        pair.m1 = actor1pc->m_material;

        auto it = m_soundLibrary.find(pair);
        if(it != m_soundLibrary.end())
        {
            std::string file = it->second;
            tbd::Resource r(file);
            std::shared_ptr<tbd::ResHandle> handle = app::g_pApp->GetCache()->GetHandle(r);

            std::shared_ptr<tbd::TransformComponent> tc0 = actor0->GetComponent<tbd::TransformComponent>(tbd::TransformComponent::COMPONENT_ID).lock();
            std::shared_ptr<tbd::TransformComponent> tc1 = actor1->GetComponent<tbd::TransformComponent>(tbd::TransformComponent::COMPONENT_ID).lock();
            util::Vec3 position = tc0->GetTransformation()->GetTranslation() + (tc1->GetTransformation()->GetTranslation() - tc0->GetTransformation()->GetTranslation()) * 0.5;
            
            FLOAT radius = 40; //Todo: make depending on force of impact
            util::ICamera* camera = app::g_pApp->GetHumanView()->GetSceneGraph()->GetCamera().get();
            util::Vec3 pos = camera->GetEyePos() - position;
            INT vol = 0;
            FLOAT l = pos.Length();

            if(radius > l)
            {
                vol = (INT)(100 * l / radius);
            }

            std::shared_ptr<proc::StaticSoundEmitterProcess> proc = std::shared_ptr<proc::StaticSoundEmitterProcess>(new proc::StaticSoundEmitterProcess(position, handle, radius, 0, vol));
            app::g_pApp->GetLogic()->AttachProcess(proc);
        }
        else
        {
            //play no sound, or a default one? I guess not...
        }
    }

    VOID SoundEngine::NewComponentDelegate(event::IEventPtr event)
    {
        std::shared_ptr<event::NewComponentCreatedEvent> pCastEventData = std::static_pointer_cast<event::NewComponentCreatedEvent>(event);
        std::shared_ptr<tbd::Actor> actor = app::g_pApp->GetLogic()->VFindActor(pCastEventData->m_actorId);

        if(pCastEventData->m_id == tbd::SoundComponent::COMPONENT_ID)
        {
            std::shared_ptr<tbd::SoundComponent> comp = actor->GetComponent<tbd::SoundComponent>(tbd::SoundComponent::COMPONENT_ID).lock();
            if(!actor->HasComponent<tbd::TransformComponent>(tbd::TransformComponent::COMPONENT_ID))
            {
                LOG_CRITICAL_ERROR("actor needs a transformcomponent");
            }
            std::shared_ptr<tbd::TransformComponent> transform = actor->GetComponent<tbd::TransformComponent>(tbd::TransformComponent::COMPONENT_ID).lock();

            tbd::Resource r(comp->m_soundFile);
            std::shared_ptr<tbd::ResHandle> handle = app::g_pApp->GetCache()->GetHandle(r);
            std::shared_ptr<proc::SoundProcess> proc;
            if(comp->m_emitter)
            {
                proc = std::shared_ptr<proc::SoundEmitterProcess>(new proc::SoundEmitterProcess(actor, transform, handle, comp->m_radius, 0, 0, comp->m_loop));
            }
            else
            {
                proc = std::shared_ptr<proc::SoundProcess>(new proc::SoundProcess(handle, 0, 100, comp->m_loop));
            }
            app::g_pApp->GetLogic()->AttachProcess(proc);
            comp->VSetHandled();
        }
    }

    SoundEngine::~SoundEngine(VOID)
    {
        m_soundLibrary.clear();
    }
}

