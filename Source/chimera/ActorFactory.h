#pragma once
#include "stdafx.h"
#include "Actor.h"

namespace chimera
{
    class ActorFactory : public IActorFactory
    {
    private:
        ActorId m_lastActorId;
        ActorId GetNextActorId(VOID) { return ++m_lastActorId; }
    protected:
        std::map<std::string, ActorComponentCreator> m_creators;
        std::map<ComponentId, ActorComponentCreator> m_creatorsId;
    public:
        ActorFactory(VOID);

        std::unique_ptr<IActor> VCreateActor(CONST CMResource& resource, std::vector<std::unique_ptr<IActor>>& actors);

        std::unique_ptr<IActor> VCreateActor(std::unique_ptr<ActorDescription> actorDesc);

        std::unique_ptr<IActorComponent> VCreateComponent(LPCSTR name);

        std::unique_ptr<IActorComponent> VCreateComponent(ComponentId id);

        std::unique_ptr<ActorDescription> VCreateActorDescription(VOID) { return std::unique_ptr<ActorDescription>(new ActorDescription(this)); }

        VOID VAddComponentCreator(ActorComponentCreator creator, LPCSTR name, ComponentId id);

        virtual ~ActorFactory(VOID) { }
    };
};

