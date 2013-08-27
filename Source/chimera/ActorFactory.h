#pragma once
#include "stdafx.h"
#include "Actor.h"

class tinyxml2::XMLElement;

namespace tbd
{
    class ActorComponent;

    typedef tbd::ActorComponent* (*ActorComponentCreator)(VOID);

    class ActorFactory
    {
        friend class _ActorDescription;
    private:
        static ActorId m_lastActorId;
        ActorId GetNextActorId(VOID) { return ++m_lastActorId; }
    protected:
        std::map<std::string, ActorComponentCreator> m_creators;
        std::map<ComponentId, ActorComponentCreator> m_creatorsId;
        virtual std::shared_ptr<tbd::ActorComponent> CreateComponent(tinyxml2::XMLElement* pData);
    public:
        ActorFactory(VOID);
        std::shared_ptr<tbd::Actor> CreateActor(CONST CHAR* ressource, std::vector<std::shared_ptr<tbd::Actor>>& actors);
        std::shared_ptr<tbd::Actor> CreateActor(tinyxml2::XMLElement* pData, std::vector<std::shared_ptr<tbd::Actor>>& actors);
        std::shared_ptr<tbd::Actor> CreateActor(ActorDescription actorDesc);
        ActorDescription CreateActorDescription(VOID) { return ActorDescription(new _ActorDescription(this)); }
        VOID ReplaceComponent(std::shared_ptr<tbd::Actor> actor, tinyxml2::XMLElement* pData);
        VOID AddComponentCreator(ActorComponentCreator creator, LPCSTR name, ComponentId id);
        virtual ~ActorFactory(VOID) { }
    };
};

