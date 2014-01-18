#pragma once
#include "CMTypes.h"

namespace chimera
{
    class IActorComponent
    {
    public:
        virtual void VSetOwner(IActor* actor) = 0;

        virtual void VCreateResources(void) = 0;

        virtual ComponentId VGetComponentId(void) const = 0;
        
        virtual IActor* VGetActor(void) = 0;
        
        virtual LPCSTR VGetName(void) const = 0;
        
        virtual void VPostInit(void) = 0;

        virtual ~IActorComponent(void) {}
    };

    class IActor
    {
    protected:
        ActorId m_id;
        std::string m_name;
    public:
        IActor(ActorId id) : m_id(id) {}

        ActorId GetId(void) { return m_id; }

        void SetName(const std::string& name)
        {
            m_name = name;
        }

        const std::string& GetName(void)
        {
            return m_name;
        }

        virtual void VAddComponent(std::unique_ptr<IActorComponent> pComponent) = 0;

        virtual IActorComponent* VGetComponent(ComponentId id) = 0;

        virtual void VQueryComponent(ComponentId id, IActorComponent** cmp) = 0;

        virtual bool VHasComponent(ComponentId id) = 0;

        virtual const std::map<ComponentId, std::unique_ptr<IActorComponent>>& VGetComponents(void) = 0;

        virtual ~IActor(void) {}
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
        virtual IActor* VCreateActor(const CMResource& resource, std::vector<std::unique_ptr<IActor>>& actors) = 0;

        virtual std::unique_ptr<IActor> VCreateActor(std::unique_ptr<ActorDescription> actorDesc) = 0;

        virtual IActor* VCreateActor(ICMStream* stream, std::vector<std::unique_ptr<IActor>>& actors) = 0;

        virtual std::unique_ptr<ActorDescription> VCreateActorDescription(void) = 0;

        virtual std::unique_ptr<IActorComponent> VCreateComponent(LPCSTR name) = 0;

        virtual std::unique_ptr<IActorComponent> VCreateComponent(ComponentId id) = 0;

        virtual std::unique_ptr<IActorComponent> VCreateComponent(ICMStream* stream) = 0;

        virtual void VAddComponentCreator(ActorComponentCreator creator, LPCSTR name, ComponentId id) = 0;

        virtual void VAddComponentSerializer(ActorComponentSerializer serializer, LPCSTR name, ComponentId id) = 0;

        virtual void VAddComponentInitializer(ActorComponentInitializer initializer, LPCSTR name, ComponentId id) = 0;

        virtual ~IActorFactory(void) {}
    };

    class ActorDescription 
    {
    private:
        std::vector<std::unique_ptr<IActorComponent>> m_components;
        IActorFactory* m_pFactory;
    public:
        ActorDescription(IActorFactory* factory) : m_pFactory(factory) {}

        template<class ComponentType>
        ComponentType* AddComponent(const std::string& comp) 
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

        std::vector<std::unique_ptr<IActorComponent>>& GetComponents(void) 
        {
            return m_components;
        }

        virtual ~ActorDescription(void) 
        {

        }
    };

    class IActorFactoryFactory
    {
    public:
        virtual IActorFactory* VCreateActorFactroy(void) = 0;
    };
}