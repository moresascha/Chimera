#include "ActorFactory.h"
#include "Components.h"
#include "Event.h"
#include "Process.h"

namespace chimera
{

    #define CM_EVENT___CC 0x12bd1c57
    class __CC : public IEvent
    {
        friend class WaitForComponentsToGethandled;
    private:
        HANDLE m_handle;
    public:
        __CC(VOID)
        {
            m_handle = CreateEvent(NULL, FALSE, FALSE, NULL);
        }

        EventType VGetEventType(VOID)
        {
            return CM_EVENT___CC;
        }

        ~__CC(VOID)
        {
            if(m_handle)
            {
                CloseHandle(m_handle);
            }
        }
    };

    class WaitForComponentsToGethandled : public RealtimeProcess
    {
    private:
        IActor* m_actor;
    public:

        WaitForComponentsToGethandled(IActor* actor) : m_actor(actor)
        {
        }

        VOID WaitForEvent(IEventPtr event)
        {
            std::shared_ptr<__CC> cc = std::static_pointer_cast<__CC>(event);
            SetEvent(cc->m_handle);
        }

        VOID VThreadProc(VOID)
        {
            /*HANDLE* pHandles = new HANDLE[m_actor->VGetComponents().size()];
            TBD_FOR_INT(m_actor->VGetComponents().size())
            {
                *(pHandles+i) = *it;
            }
            
            WaitForMultipleObjects((DWORD)handles.size(), pHandles, TRUE, INFINITE);*/
            
            QUEUE_EVENT_TSAVE(new ActorCreatedEvent(m_actor->GetId()));
            
            //SAFE_ARRAY_DELETE(pHandles);

            Succeed();
        }
    };

    class CreateActorComponentsProcess : public RealtimeProcess 
    {
    private:
        IActor* m_actor;
        std::string m_name;
    public:
        CreateActorComponentsProcess(IActor* actor) : m_actor(actor) 
        {
            std::unique_ptr<IProcess> proc(new WaitForComponentsToGethandled(actor));
            VSetChild(std::move(proc));
            SetPriority(THREAD_PRIORITY_HIGHEST);
        }

        VOID VThreadProc(VOID)
        {
            for(auto it = m_actor->VGetComponents().begin(); it != m_actor->VGetComponents().end(); ++it)
            {
                it->second->VCreateResources();
            }
            for(auto it = m_actor->VGetComponents().begin(); it != m_actor->VGetComponents().end(); ++it)
            {
                it->second->VPostInit();
                QUEUE_EVENT_TSAVE(new NewComponentCreatedEvent(it->second->VGetComponentId(), m_actor->GetId()));
            }
            Succeed();
        }
    };

    IActorComponent* CreateTransformComponent(VOID) 
    {
        return new TransformComponent;
    }

    IActorComponent* CreateRenderComponent(VOID)
    {
        return new RenderComponent;
    }

    IActorComponent* CreateCameraComponent(VOID) 
    {
        return new CameraComponent;
    }

    IActorComponent* CreatePhysicComponent(VOID) 
    {
        return new PhysicComponent;
    }

    IActorComponent* CreateLightComponent(VOID) 
    {
        return new LightComponent;
    }

    IActorComponent* CreatePickableComponent(VOID)
    {
        return new PickableComponent;
    }

    IActorComponent* CreateSoundEmitterComponent(VOID)
    {
        return new SoundComponent;
    }

    IActorComponent* CreateParentComponent(VOID)
    {
        return new ParentComponent;
    }

    ActorFactory::ActorFactory(VOID) : m_lastActorId(0)
    {
        VAddComponentCreator(CreateTransformComponent, "TransformComponent", CM_CMP_TRANSFORM);
        VAddComponentCreator(CreateRenderComponent, "RenderComponent", CM_CMP_RENDERING);
        VAddComponentCreator(CreateCameraComponent, "CameraComponent", CM_CMP_CAMERA);
        VAddComponentCreator(CreatePhysicComponent, "PhysicComponent", CM_CMP_PHX);
        VAddComponentCreator(CreateLightComponent, "LightComponent", CM_CMP_LIGHT);
        VAddComponentCreator(CreateSoundEmitterComponent, "SoundComponent", CM_CMP_SOUND);
        VAddComponentCreator(CreateParentComponent, "ParentComponent", CM_CMP_PARENT_ACTOR);
    }

    VOID ActorFactory::VAddComponentCreator(ActorComponentCreator creator, LPCSTR name, ComponentId id)
    {
        m_creators[std::string(name)] = creator;
        m_creatorsId[id] = creator;
    }

    std::unique_ptr<IActor> ActorFactory::VCreateActor(std::unique_ptr<ActorDescription> desc)
    {
        std::unique_ptr<IActor> actor(new Actor(GetNextActorId()));
        for(auto it = desc->GetComponents().begin(); it != desc->GetComponents().end(); ++it)
        {
            (*it)->VSetOwner(actor.get());
            actor->VAddComponent(std::move(*it));
        }
        std::unique_ptr<IProcess> proc = std::unique_ptr<IProcess>(new CreateActorComponentsProcess(actor.get()));
        CmGetApp()->VGetLogic()->VGetProcessManager()->VAttachWithScheduler(std::move(proc));
        return actor; 
    }


    std::unique_ptr<IActorComponent> ActorFactory::VCreateComponent(LPCSTR name)
    {
        auto it = m_creators.find(name);
        if(it == m_creators.end())
        {
            LOG_CRITICAL_ERROR_A("Component for '%s' creator does not exists!", name);
        }

        return std::unique_ptr<IActorComponent>(it->second());
    }

    std::unique_ptr<IActorComponent> ActorFactory::VCreateComponent(ComponentId id)
    {
        auto it = m_creatorsId.find(id);
        if(it == m_creatorsId.end())
        {
            LOG_CRITICAL_ERROR_A("Component for '%d' creator does not exists!", id);
        }

        return std::unique_ptr<IActorComponent>(it->second());
    }

    std::unique_ptr<IActor> ActorFactory::VCreateActor(CONST CMResource& resource, std::vector<std::unique_ptr<IActor>>& actors)
    {
        return NULL;
    }
    /*
    std::shared_ptr<chimera::Actor> ActorFactory::VCreateActor(tinyxml2::XMLElement* pData, std::vector<std::shared_ptr<chimera::Actor>>& actors) 
    {
        if(!pData)
        {
            LOG_CRITICAL_ERROR("Failed to load XML file");
            return NULL;
        }

        std::shared_ptr<chimera::Actor> pActor(new Actor(GetNextActorId()));

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
                std::shared_ptr<chimera::Actor> child = CreateActor(pNode, actors);
                std::shared_ptr<chimera::ParentComponent> pcmp = std::shared_ptr<chimera::ParentComponent>(new chimera::ParentComponent());
                pcmp->m_parentId = pActor->GetId();
                child->AddComponent(pcmp);
                pcmp->SetOwner(child);
            }
            else
            {
                std::shared_ptr<chimera::ActorComponent> pComponent(CreateComponent(pNode));
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
        std::shared_ptr<chimera::Process> chimera = std::shared_ptr<chimera::Process>(new CreateActorComponentsProcess(pActor));
        chimera::g_pApp->GetLogic()->GetProcessManager()->VAttachWithScheduler(chimera);
        pActor->PostInit();
        actors.push_back(pActor);
        return pActor;
    } */
    /*
    std::shared_ptr<chimera::Actor> ActorFactory::VCreateActor(CONST CHAR* ressource, std::vector<std::shared_ptr<chimera::Actor>>& actors) 
    {
        tinyxml2::XMLDocument doc;
        chimera::CMResource r(ressource);
        std::shared_ptr<chimera::ResHandle> handle = chimera::g_pApp->GetCache()->GetHandle(r);
        doc.Parse(handle->Buffer());
        tinyxml2::XMLElement* root = doc.RootElement();
        return CreateActor(root, actors);
    }

    std::shared_ptr<chimera::ActorComponent> ActorFactory::VCreateComponent(tinyxml2::XMLElement* pData) 
    {

        std::shared_ptr<chimera::ActorComponent> pComponent;
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

    VOID ActorFactory::VReplaceComponent(std::shared_ptr<chimera::Actor> actor, tinyxml2::XMLElement* pData) 
    {
        std::shared_ptr<chimera::ActorComponent> pComponent(CreateComponent(pData));
        if(pComponent)
        {
            actor->ReplaceComponent(pComponent);

            pComponent->SetOwner(actor);
        }
    }*/
}
