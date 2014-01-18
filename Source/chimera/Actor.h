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

      virtual ~Actor(void) 
      {
          TBD_FOR(m_components)
          {
              it->second.reset();
          }
      }

      const std::map<ComponentId, std::unique_ptr<IActorComponent>>& VGetComponents(void) { return m_components; }

      template <typename T>
      T* VGetComponent(ComponentId id) 
      {
          return (T*)VGetComponent(id);
      }

      IActorComponent* VGetComponent(ComponentId id) 
      {
          auto it = m_components.find(id);

          if(it != m_components.end())
          {
              return it->second.get();
          }
          return NULL;
      }

      void VQueryComponent(ComponentId id, IActorComponent** cmp)
      {
          *cmp = VGetComponent(id);
      }

      void VAddComponent(std::unique_ptr<IActorComponent> pComponent)
      {
          if(m_components.find(pComponent->VGetComponentId()) != m_components.end())
          {
              LOG_CRITICAL_ERROR_A("%s", "Replacing an AC!");
          }
          m_components.insert(std::pair<ComponentId, std::unique_ptr<IActorComponent>>(pComponent->VGetComponentId(), std::move(pComponent)));
      }

      bool VHasComponent(ComponentId id)
      {
          return VGetComponent(id) != NULL;
      }
  };
};
