#include "ActorFactory.h"
#include "Thread.h"
#include "Process.h"
#include "Event.h"
#include "tinyxml2.h"
#include "EventManager.h"
#include "GameApp.h"
#include "ProcessManager.h"
#include "GameLogic.h"
#include "Components.h"
#include "Event.h"
namespace tbd
{

    class WaitForComponentsToGethandled : public proc::RealtimeProcess
    {
    private:
        std::shared_ptr<tbd::Actor> m_actor;
    public:
        WaitForComponentsToGethandled(std::shared_ptr<tbd::Actor> actor) : m_actor(actor)
        {
        }

        BOOL Done(VOID)
        {
            for(auto it = m_actor->GetComponents().begin(); it != m_actor->GetComponents().end(); ++it)
            {
                if(!it->second->IsWaitTillHandled())
                {
                    return FALSE;
                }
            }
            return TRUE;
        }

        VOID VThreadProc(VOID)
        {
            /*while(!Done()) 
            {
                Sleep(60);
            } */
            std::list<HANDLE> handles;

            for(auto it = m_actor->GetComponents().begin(); it != m_actor->GetComponents().end(); ++it)
            {
                if(it->second->IsWaitTillHandled())
                {
                    handles.push_back(it->second->GetHandle());
                }
            }
            
            HANDLE* pHandles = new HANDLE[handles.size()];
            INT pos = 0;
            TBD_FOR(handles)
            {
                *(pHandles+pos++) = *it;
            }
            
            WaitForMultipleObjects((DWORD)handles.size(), pHandles, TRUE, INFINITE);
            
            event::IEventPtr actorCreatedEvent(new event::ActorCreatedEvent(m_actor->GetId()));
            event::IEventManager::Get()->VQueueEventThreadSave(actorCreatedEvent);
            
            SAFE_ARRAY_DELETE(pHandles);

            Succeed();
        }
    };

    class CreateActorComponentsProcess : public proc::RealtimeProcess 
    {
    private:
        std::shared_ptr<tbd::Actor> m_actor;
        std::string m_name;
    public:
        CreateActorComponentsProcess(std::shared_ptr<tbd::Actor> actor) : m_actor(actor) 
        {
            std::shared_ptr<WaitForComponentsToGethandled> proc = std::shared_ptr<WaitForComponentsToGethandled>(new WaitForComponentsToGethandled(actor));
            VSetChild(proc);
            SetPriority(THREAD_PRIORITY_HIGHEST);
        }

        VOID VThreadProc(VOID)
        {
            for(auto it = m_actor->GetComponents().begin(); it != m_actor->GetComponents().end(); ++it)
            {
                it->second->VCreateResources();
            }
            for(auto it = m_actor->GetComponents().begin(); it != m_actor->GetComponents().end(); ++it)
            {
                it->second->VPostInit();
            }
            Succeed();
        }
    };

    ActorComponent* CreateTransformComponent(VOID) 
    {
        return new TransformComponent;
    }

    ActorComponent* CreateRenderComponent(VOID)
    {
        return new RenderComponent;
    }

    ActorComponent* CreateCameraComponent(VOID) 
    {
        return new CameraComponent;
    }

    ActorComponent* CreatePhysicComponent(VOID) 
    {
        return new PhysicComponent;
    }

    ActorComponent* CreateLightComponent(VOID) 
    {
        return new LightComponent;
    }

    ActorComponent* CreatePickableComponent(VOID)
    {
        return new PickableComponent;
    }

    ActorComponent* CreateParticleComponent(VOID)
    {
        return new ParticleComponent;
    }

    ActorComponent* CreateSoundEmitterComponent(VOID)
    {
        return new SoundComponent;
    }

    ActorComponent* CreateParentComponent(VOID)
    {
        return new ParentComponent;
    }

    ActorId ActorFactory::m_lastActorId = 0;

    ActorFactory::ActorFactory(VOID) 
    {
        m_creators["TransformComponent"] = CreateTransformComponent;
        m_creatorsId[TransformComponent::COMPONENT_ID] = CreateTransformComponent;

        m_creators["RenderComponent"] = CreateRenderComponent;
        m_creatorsId[RenderComponent::COMPONENT_ID] = CreateRenderComponent;

        m_creators["CameraComponent"] = CreateCameraComponent;
        m_creatorsId[CameraComponent::COMPONENT_ID] = CreateCameraComponent;

        m_creators["PhysicComponent"] = CreatePhysicComponent;
        m_creatorsId[PhysicComponent::COMPONENT_ID] = CreatePhysicComponent;

        m_creators["LightComponent"] = CreateLightComponent;
        m_creatorsId[LightComponent::COMPONENT_ID] = CreateLightComponent;

        m_creators["PickableComponent"] = CreatePickableComponent;
        m_creatorsId[PickableComponent::COMPONENT_ID] = CreatePickableComponent;

        m_creators["ParticleComponent"] = CreateParticleComponent;
        m_creatorsId[ParticleComponent::COMPONENT_ID] = CreateParticleComponent;

        m_creators["SoundComponent"] = CreateSoundEmitterComponent;
        m_creatorsId[SoundComponent::COMPONENT_ID] = CreateSoundEmitterComponent;

        m_creators["ParentComponent"] = CreateParentComponent;
        m_creatorsId[ParentComponent::COMPONENT_ID] = CreateParentComponent;
    }

    VOID ActorFactory::AddComponentCreator(ActorComponentCreator creator, LPCSTR name, ComponentId id)
    {
        m_creators[std::string(name)] = creator;
        m_creatorsId[id] = creator;
    }

    std::shared_ptr<tbd::Actor> ActorFactory::CreateActor(ActorDescription desc)
    {
        std::shared_ptr<tbd::Actor> actor(new Actor(GetNextActorId()));
        for(auto it = desc->GetComponents()->begin(); it != desc->GetComponents()->end(); ++it)
        {
            it->get()->SetOwner(actor);
            actor->AddComponent(*it);
        }
        std::shared_ptr<proc::Process> proc = std::shared_ptr<proc::Process>(new CreateActorComponentsProcess(actor));
        app::g_pApp->GetLogic()->GetProcessManager()->AttachWithScheduler(proc);
        actor->PostInit();
        return actor; 
    }

    std::shared_ptr<tbd::Actor> ActorFactory::CreateActor(tinyxml2::XMLElement* pData, std::vector<std::shared_ptr<tbd::Actor>>& actors) 
    {
        if(!pData)
        {
            LOG_CRITICAL_ERROR("Failed to load XML file");
            return NULL;
        }

        std::shared_ptr<tbd::Actor> pActor(new Actor(GetNextActorId()));

        if(!pActor->Init(pData))
        {
            LOG_CRITICAL_ERROR("Failed to init actor");
            return NULL;
        }

        for(tinyxml2::XMLElement* pNode = pData->FirstChildElement(); pNode; pNode = pNode->NextSiblingElement())
        {
            std::string name(pNode->Value());
            if(name == "Actor")
            {
                std::shared_ptr<tbd::Actor> child = CreateActor(pNode, actors);
                std::shared_ptr<tbd::ParentComponent> pcmp = std::shared_ptr<tbd::ParentComponent>(new tbd::ParentComponent());
                pcmp->m_parentId = pActor->GetId();
                child->AddComponent(pcmp);
                pcmp->SetOwner(child);
            }
            else
            {
                std::shared_ptr<tbd::ActorComponent> pComponent(CreateComponent(pNode));
                if(pComponent)
                {
                    pActor->AddComponent(pComponent);

                    pComponent->SetOwner(pActor);
                }
                else
                {
                    return NULL;
                }
            }
        }
        std::shared_ptr<proc::Process> proc = std::shared_ptr<proc::Process>(new CreateActorComponentsProcess(pActor));
        app::g_pApp->GetLogic()->GetProcessManager()->AttachWithScheduler(proc);
        pActor->PostInit();
        actors.push_back(pActor);
        return pActor;
    }

    std::shared_ptr<tbd::Actor> ActorFactory::CreateActor(CONST CHAR* ressource, std::vector<std::shared_ptr<tbd::Actor>>& actors) 
    {
        tinyxml2::XMLDocument doc;
        tbd::Resource r(ressource);
        std::shared_ptr<tbd::ResHandle> handle = app::g_pApp->GetCache()->GetHandle(r);
        doc.Parse(handle->Buffer());
        tinyxml2::XMLElement* root = doc.RootElement();
        return CreateActor(root, actors);
    }

    std::shared_ptr<tbd::ActorComponent> ActorFactory::CreateComponent(tinyxml2::XMLElement* pData) 
    {

        std::shared_ptr<tbd::ActorComponent> pComponent;
        std::string name(pData->Value());
        auto it = m_creators.find(name);
        if(it != m_creators.end())
        {
            ActorComponentCreator creator = it->second;
            pComponent.reset(creator());
        }
        else
        {
            LOG_CRITICAL_ERROR_A("Could not find actor component: %s", name.c_str());
        }

        if(pComponent)
        {
            if(!pComponent->VInit(pData))
            {
                LOG_CRITICAL_ERROR_A("Failed to init actor component: %s", name.c_str());
            }
        }
        return pComponent;
    }

    VOID ActorFactory::ReplaceComponent(std::shared_ptr<tbd::Actor> actor, tinyxml2::XMLElement* pData) 
    {
        std::shared_ptr<tbd::ActorComponent> pComponent(CreateComponent(pData));
        if(pComponent)
        {
            actor->ReplaceComponent(pComponent);

            pComponent->SetOwner(actor);
        }
    }
}
