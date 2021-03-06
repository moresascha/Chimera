#include "ActorFactory.h"
#include "Components.h"
#include "Event.h"
#include "Process.h"
#include "tinyxml2.h"

namespace chimera
{

    #define CM_EVENT___CC 0x12bd1c57
    class __CC : public IEvent
    {
        friend class WaitForComponentsToGethandled;
    private:
        HANDLE m_handle;
    public:
        __CC(void)
        {
            m_handle = CreateEvent(NULL, false, false, NULL);
        }

        EventType VGetEventType(void)
        {
            return CM_EVENT___CC;
        }

        ~__CC(void)
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

        void WaitForEvent(IEventPtr event)
        {
            std::shared_ptr<__CC> cc = std::static_pointer_cast<__CC>(event);
            SetEvent(cc->m_handle);
        }

        void VThreadProc(void)
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

        void VThreadProc(void)
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

    IActorComponent* CreateTransformComponent(void) 
    {
        return new TransformComponent;
    }

    IActorComponent* CreateRenderComponent(void)
    {
        return new RenderComponent;
    }

    IActorComponent* CreateCameraComponent(void) 
    {
        return new CameraComponent;
    }

    IActorComponent* CreatePhysicComponent(void) 
    {
        return new PhysicComponent;
    }

    IActorComponent* CreateLightComponent(void) 
    {
        return new LightComponent;
    }

    IActorComponent* CreatePickableComponent(void)
    {
        return new PickableComponent;
    }

    IActorComponent* CreateSoundEmitterComponent(void)
    {
        return new SoundComponent;
    }

    IActorComponent* CreateParentComponent(void)
    {
        return new ParentComponent;
    }

    IActorComponent* CreateControllerComponent(void)
    {
        return new ControllerComponent;
    }

    IActorComponent* CreateParticleComponent(void)
    {
        return new ParticleComponent;
    }

    bool InitializeTransformComponent(IActorComponent* cmp, ICMStream* stream);

    bool InitializeRenderingComponent(IActorComponent* cmp, ICMStream* stream);

    bool InitializeCameraComponent(IActorComponent* cmp, ICMStream* stream);

    bool InitializeControllerComponent(IActorComponent* cmp, ICMStream* stream);

    bool InitializePhysicsComponent(IActorComponent* cmp, ICMStream* stream);

    bool InitializeLightComponent(IActorComponent* cmp, ICMStream* stream);

    bool InitializeParticleComponent(IActorComponent* cmp, ICMStream* stream);

    bool InitializeNothing(IActorComponent* cmp, ICMStream* stream)
    {

        return true;
    }

    ActorFactory::ActorFactory(void) : m_lastActorId(CM_INVALID_ACTOR_ID + 1)
    {
        VAddComponentCreator(CreateTransformComponent, "TransformComponent", CM_CMP_TRANSFORM);
        VAddComponentCreator(CreateRenderComponent, "RenderComponent", CM_CMP_RENDERING);
        VAddComponentCreator(CreateCameraComponent, "CameraComponent", CM_CMP_CAMERA);
        VAddComponentCreator(CreatePhysicComponent, "PhysicComponent", CM_CMP_PHX);
        VAddComponentCreator(CreateLightComponent, "LightComponent", CM_CMP_LIGHT);
        VAddComponentCreator(CreateSoundEmitterComponent, "SoundComponent", CM_CMP_SOUND);
        VAddComponentCreator(CreateParentComponent, "ParentComponent", CM_CMP_PARENT_ACTOR);
        VAddComponentCreator(CreateControllerComponent, "ControllerComponent", CM_CMP_CONTROLLER);
        VAddComponentCreator(CreatePickableComponent, "PickableComponent", CM_CMP_PICKABLE);
        VAddComponentCreator(CreateParticleComponent, "ParticleComponent", CM_CMP_PARTICLE);

        VAddComponentInitializer(InitializeTransformComponent, "TransformComponent", CM_CMP_TRANSFORM);
        VAddComponentInitializer(InitializeRenderingComponent, "RenderComponent", CM_CMP_RENDERING);
        VAddComponentInitializer(InitializeCameraComponent, "CameraComponent", CM_CMP_CAMERA);
        VAddComponentInitializer(InitializeControllerComponent, "ControllerComponent", CM_CMP_CONTROLLER);
        VAddComponentInitializer(InitializePhysicsComponent, "PhysicComponent", CM_CMP_PHX);
        VAddComponentInitializer(InitializeLightComponent, "LightComponent", CM_CMP_LIGHT);
        VAddComponentInitializer(InitializeParticleComponent, "ParticleComponent", CM_CMP_PARTICLE);
    }

    void ActorFactory::VAddComponentCreator(ActorComponentCreator creator, LPCSTR name, ComponentId id)
    {
        m_creators[std::string(name)] = creator;
        m_creatorsId[id] = creator;
    }

    void ActorFactory::VAddComponentSerializer(ActorComponentSerializer serializer, LPCSTR name, ComponentId id)
    {
        m_serializerId[std::string(name)] = serializer;
    }

    void ActorFactory::VAddComponentInitializer(ActorComponentInitializer initializer, LPCSTR name, ComponentId id)
    {
        m_initializerId[std::string(name)] = initializer;
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

    IActor* ActorFactory::VCreateActor(ICMStream* stream, std::vector<std::unique_ptr<IActor>>& actors)
    {
        IActor* pActor = new Actor(GetNextActorId());
        tinyxml2::XMLNode* pData = (tinyxml2::XMLNode*)stream;

        if(pData->ToElement()->Attribute("name"))
        {
            std::string name = pData->ToElement()->Attribute("name");
            pActor->SetName(name);
        }    

        for(tinyxml2::XMLElement* pNode = pData->FirstChildElement(); pNode; pNode = pNode->NextSiblingElement())
        {
            std::string name(pNode->Value());
            if(name == "Actor")
            {
                IActor* child = NULL;
                if(pNode->Attribute("file"))
                {
                    CMResource r = pNode->Attribute("file");
                    child = VCreateActor(r, actors);
   
                }
                else
                {
                    child = VCreateActor((ICMStream*)pNode, actors);
                }

                std::unique_ptr<ParentComponent> pcmp = std::unique_ptr<ParentComponent>(new ParentComponent());
                pcmp->m_parentId = pActor->GetId();
                if(pNode->Attribute("name"))
                {
                    child->SetName(std::string(pNode->Attribute("name")));
                }
                pcmp->VSetOwner(child);
                child->VAddComponent(std::move(pcmp));
            }
            else
            {
                std::string componentName(pNode->Value());

                std::unique_ptr<IActorComponent> pComponent(VCreateComponent((ICMStream*)pNode));

                if(pComponent)
                {
                    IActorComponent* cmp = pComponent.get();
                    pComponent->VSetOwner(pActor);
                    pActor->VAddComponent(std::move(pComponent));
                    auto itInit = m_initializerId.find(componentName);
                    if(itInit != m_initializerId.end())
                    {
                        ActorComponentInitializer initializer = itInit->second;
                        initializer(cmp, (ICMStream*)pNode);
                    }
                }   
            }
        }

        std::unique_ptr<IProcess> proc = std::unique_ptr<IProcess>(new CreateActorComponentsProcess(pActor));
        CmGetApp()->VGetLogic()->VGetProcessManager()->VAttachWithScheduler(std::move(proc));
        actors.push_back(std::unique_ptr<IActor>(pActor));
        return pActor;
    }

    IActor* ActorFactory::VCreateActor(const CMResource& resource, std::vector<std::unique_ptr<IActor>>& actors)
    {
        tinyxml2::XMLDocument doc;
        CMResource r(CmGetApp()->VGetConfig()->VGetString("sActorPath") + resource);
        std::shared_ptr<IResHandle> handle = CmGetApp()->VGetCache()->VGetHandle(r);
        doc.Parse(handle->VBuffer());
        tinyxml2::XMLNode* pData = doc.FirstChild();

        if(!pData)
        {
            LOG_CRITICAL_ERROR_A("Failed to load XML file: %s, %s", doc.GetErrorStr1(), doc.GetErrorStr2());
        }

        return VCreateActor((ICMStream*)pData, actors);
    }

    std::unique_ptr<IActorComponent> ActorFactory::VCreateComponent(ICMStream* stream) 
    {
        tinyxml2::XMLElement* pData = (tinyxml2::XMLElement*)stream;
        IActorComponent* pComponent;
        std::string name(pData->Value());
        auto it = m_creators.find(name);
        if(it != m_creators.end())
        {
            ActorComponentCreator creator = it->second;
            pComponent = creator();
        }
        else
        {
            LOG_CRITICAL_ERROR_A("Could not find actor component: %s", name.c_str());
        }
        return std::unique_ptr<IActorComponent>(pComponent);
    }

    /*
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
