#pragma once
#include "stdafx.h"
#include "Actor.h"

namespace chimera
{
    class ActorFactory : public IActorFactory
    {
    private:
        ActorId m_lastActorId;
        ActorId GetNextActorId(void) { return ++m_lastActorId; }
    protected:
        std::map<std::string, ActorComponentCreator> m_creators;
        std::map<ComponentId, ActorComponentCreator> m_creatorsId;
        std::map<std::string, ActorComponentSerializer> m_serializerId;
        std::map<std::string, ActorComponentInitializer> m_initializerId;

    public:
        ActorFactory(void);

        std::unique_ptr<IActorComponent> VCreateComponent(ICMStream* stream);

        IActor* VCreateActor(const CMResource& resource, std::vector<std::unique_ptr<IActor>>& actors);

        IActor* VCreateActor(ICMStream* stream, std::vector<std::unique_ptr<IActor>>& actors);

        std::unique_ptr<IActor> VCreateActor(std::unique_ptr<ActorDescription> actorDesc);

        std::unique_ptr<IActorComponent> VCreateComponent(LPCSTR name);

        std::unique_ptr<IActorComponent> VCreateComponent(ComponentId id);

        std::unique_ptr<ActorDescription> VCreateActorDescription(void) { return std::unique_ptr<ActorDescription>(new ActorDescription(this)); }

        void VAddComponentCreator(ActorComponentCreator creator, LPCSTR name, ComponentId id);

        void VAddComponentSerializer(ActorComponentSerializer serializer, LPCSTR name, ComponentId id);

        void VAddComponentInitializer(ActorComponentInitializer serializer, LPCSTR name, ComponentId id);

        virtual ~ActorFactory(void) { }
    };
};

