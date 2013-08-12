#include "Actor.h"
#include "Components.h"

namespace tbd 
{

    Actor::Actor(ActorId id) : m_id(id) 
    {
        std::stringstream ss;
        ss << "Actor_";
        ss << m_id;
        SetName(ss.str());
    }

    VOID Actor::PostInit(VOID)
    {

    }

    VOID Actor::Update(ULONG millis)
    {
        for(auto it = this->m_components.begin(); it != this->m_components.end(); ++it)
        {
            it->second->VUpdate(millis);
        }
    }

    VOID Actor::Destroy(VOID) 
    {
        for(auto it = this->m_components.begin(); it != this->m_components.end(); ++it)
        {
            it->second->VDestroy();
        }
    }

    VOID Actor::ReplaceComponent(std::shared_ptr<tbd::ActorComponent> pComponent) 
    {
        auto it = this->m_components.find(pComponent->GetComponentId());
        if(it == this->m_components.end())
        {
            LOG_WARNING("Nothing to replace");
            return;
        }
        it->second->VDestroy();
        this->m_components.erase(it);
        this->m_components[pComponent->GetComponentId()] = pComponent;
    }

    VOID Actor::AddComponent(std::shared_ptr<tbd::ActorComponent> pComponent) 
    {
        auto it = this->m_components.find(pComponent->GetComponentId());
        if(it != this->m_components.end())
        {
            LOG_WARNING("trying to replace an actor component");
        }
        else
        {
            this->m_components[pComponent->GetComponentId()] = pComponent;
        }
    }
}