#pragma once
#include "stdafx.h"

namespace chimera 
{
  class Actor : public IActor
  {
  private:
      std::map<ComponentId, std::unique_ptr<IActorComponent>> m_components;
  public:
      Actor(ActorId id);

      virtual ~Actor(VOID) 
	  {
		  TBD_FOR(m_components)
		  {
			  it->second.reset();
		  }
	  }

      CONST std::map<ComponentId, std::unique_ptr<IActorComponent>>& VGetComponents(VOID) { return m_components; } 

      IActorComponent* VGetComponent(ComponentId id) 
      {
          auto it = m_components.find(id);

          if(it != m_components.end())
          {
              return it->second.get();
          }
          return NULL;
      }

      VOID VAddComponent(std::unique_ptr<IActorComponent> pComponent)
      {
          if(m_components.find(pComponent->VGetComponentId()) != m_components.end())
          {
              LOG_CRITICAL_ERROR_A("%s", "Replacing an AC!");
          }
          m_components.insert(std::pair<ComponentId, std::unique_ptr<IActorComponent>>(pComponent->VGetComponentId(), std::move(pComponent)));
      }

      BOOL VHasComponent(ComponentId id)
      {
          return VGetComponent(id) != NULL;
      }
  };
};
