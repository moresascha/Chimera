#pragma once
#include "CMTypes.h"

namespace chimera
{
    class IActorComponent
    {
    public:
        virtual VOID VSetOwner(IActor* actor) = 0;

        virtual VOID VCreateResources(VOID) = 0;

        virtual VOID VSerialize(IStream* stream) CONST = 0;
        
        virtual BOOL VInitialize(IStream* stream) = 0;

        virtual ComponentId VGetComponentId(VOID) CONST = 0;
        
        virtual IActor* VGetActor(VOID) = 0;
        
        virtual LPCSTR VGetName(VOID) CONST = 0;
        
        virtual VOID VPostInit(VOID) = 0;

        virtual ~IActorComponent(VOID) {}
    };

    class IActor
    {
    protected:
        ActorId m_id;
        std::string m_name;
    public:
        IActor(ActorId id) : m_id(id) {}

        ActorId GetId(VOID) { return m_id; }

        VOID SetName(CONST std::string& name)
        {
            m_name = name;
        }

        CONST std::string& GetName(VOID)
        {
            return m_name;
        }

        virtual VOID VAddComponent(std::unique_ptr<IActorComponent> pComponent) = 0;

        virtual IActorComponent* VGetComponent(ComponentId id) = 0;

        virtual BOOL VHasComponent(ComponentId id) = 0;

        virtual CONST std::map<ComponentId, std::unique_ptr<IActorComponent>>& VGetComponents(VOID) = 0;

        virtual ~IActor(VOID) {}
    };

    template <typename T>
    T* GetActorCompnent(IActor* actor, ComponentId id)
    {
        IActorComponent* cmp = actor->VGetComponent(id);
        if(cmp)
        {
            return (T*)(cmp);
        }
        return NULL;
    }
    
    class IActorFactory
    {
    public:
        virtual std::unique_ptr<IActor> VCreateActor(CONST CMResource& resource, std::vector<std::unique_ptr<IActor>>&) = 0;

        virtual std::unique_ptr<IActor> VCreateActor(std::unique_ptr<ActorDescription> actorDesc) = 0;

        virtual std::unique_ptr<ActorDescription> VCreateActorDescription(VOID) = 0;

        virtual std::unique_ptr<IActorComponent> VCreateComponent(LPCSTR name) = 0;

        virtual std::unique_ptr<IActorComponent> VCreateComponent(ComponentId id) = 0;

        virtual VOID VAddComponentCreator(ActorComponentCreator creator, LPCSTR name, ComponentId id) = 0;

        virtual ~IActorFactory(VOID) {}
    };

    class ActorDescription 
    {
    private:
        std::vector<std::unique_ptr<IActorComponent>> m_components;
        IActorFactory* m_pFactory;
    public:
        ActorDescription(IActorFactory* factory) : m_pFactory(factory) {}

        template<class ComponentType>
        ComponentType* AddComponent(CONST std::string& comp) 
        {
            std::unique_ptr<IActorComponent> cmp = m_pFactory->VCreateComponent(comp.c_str());
            if(!cmp)
            {
                LOG_CRITICAL_ERROR_A("%s does not exist", cmp->VGetName());
            }
            m_components.push_back(std::move(cmp));
            return (ComponentType*)m_components.back().get();
        }

        template<class ComponentType>
        ComponentType* AddComponent(ComponentId id) 
        {
            std::unique_ptr<IActorComponent> cmp = m_pFactory->VCreateComponent(id);
            if(!cmp)
            {
                LOG_CRITICAL_ERROR_A("%d does not exist", id);
            }
            m_components.push_back(std::move(cmp));
            return (ComponentType*)m_components.back().get();
        }

        std::vector<std::unique_ptr<IActorComponent>>& GetComponents(VOID) 
        {
            return m_components;
        }

        virtual ~ActorDescription(VOID) 
        {

        }
    };

    class IActorFactoryFactory
    {
    public:
        virtual IActorFactory* VCreateActorFactroy(VOID) = 0;
    };
}