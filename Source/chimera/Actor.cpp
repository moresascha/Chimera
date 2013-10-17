#include "Actor.h"

namespace chimera 
{
    Actor::Actor(ActorId id) : IActor(id) 
    {
        std::stringstream ss;
        ss << "Actor_";
        ss << m_id;
        SetName(ss.str());
    }

    /*VOID Actor::VReplaceComponent(std::unique_ptr<IActorComponent> pComponent) 
    {
        auto it = m_components.find(pComponent->VGetComponentId());
        if(it == this->m_components.end())
        {
            LOG_WARNING("Nothing to replace");
            return;
        }
        it->second->VDestroy();
        this->m_components.erase(it);
        this->m_components[pComponent->VGetComponentId()] = pComponent;
    } */
}