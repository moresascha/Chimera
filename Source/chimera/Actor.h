#pragma once
#include "stdafx.h"

#define INVALID_ACTOR_ID 0
typedef ULONG ActorId;
typedef UINT ComponentId;

namespace tinyxml2
{
    class XMLElement;
}

namespace tbd 
{

  class ActorComponent;
  class Actor;
  class _ActorDescription {
      friend class ActorFactory;
  private:    
      std::vector<std::shared_ptr<tbd::ActorComponent>> m_components;
      ActorFactory* m_pFactory;
      _ActorDescription(ActorFactory* factory) : m_pFactory(factory) {}

  public:
      template<class ComponentType>
      ComponentType* AddComponent(std::string comp) 
      {
          auto it = m_pFactory->m_creators.find(comp);
          if(it == m_pFactory->m_creators.end())
          {
              LOG_CRITICAL_ERROR_A("%s does not exist", comp.c_str());
          }
          std::shared_ptr<tbd::ActorComponent>& acomp = std::shared_ptr<tbd::ActorComponent>((it->second)());
          m_components.push_back(acomp);

          //std::shared_ptr<ComponentType> pSub();
          return (ComponentType*)acomp.get();//std::tr1::static_pointer_cast<ComponentType>(acomp);
      }

      template<class ComponentType>
      ComponentType* AddComponent(CONST ComponentId id) 
      {
          auto it = m_pFactory->m_creatorsId.find(id);
          if(it == m_pFactory->m_creatorsId.end())
          {
              LOG_CRITICAL_ERROR("component does not exist");
          }
          std::shared_ptr<tbd::ActorComponent>& acomp = std::shared_ptr<tbd::ActorComponent>((it->second)());
          m_components.push_back(acomp);

          //std::shared_ptr<ComponentType> pSub(std::tr1::static_pointer_cast<ComponentType>(acomp));
          return (ComponentType*)acomp.get();//std::tr1::static_pointer_cast<ComponentType>(acomp);//pSub;
      }

      std::vector<std::shared_ptr<tbd::ActorComponent>>* GetComponents(VOID) {
          return &m_components;
      }
  };

  typedef std::shared_ptr<_ActorDescription> ActorDescription;

  class Actor 
  {
      friend class ActorFactory;
  private:
      std::map<ComponentId, std::shared_ptr<tbd::ActorComponent>> m_components;
      ActorId m_id;
      VOID AddComponent(std::shared_ptr<tbd::ActorComponent> pComponent);
      VOID ReplaceComponent(std::shared_ptr<tbd::ActorComponent> pComponent);
      std::string m_name;

  public:
      Actor(ActorId id);
      ~Actor(VOID) {}

      BOOL Init(tinyxml2::XMLElement* pData) { return TRUE; } //for what?
      VOID PostInit(VOID);
      VOID Destroy(VOID);
      VOID Update(ULONG millis);
      ActorId GetId(VOID) CONST { return m_id; }

      VOID SetName(CONST std::string& name)
      {
          m_name = name;
      }

      CONST std::string& GetName(VOID)
      {
          return m_name;
      }

      std::map<ComponentId, std::shared_ptr<tbd::ActorComponent>>& GetComponents(VOID) { return m_components; } 

      template<class ComponentType>
      std::weak_ptr<ComponentType> GetComponent(CONST ComponentId id) {

          auto it = m_components.find(id);

          if(it != m_components.end())
          {
              std::shared_ptr<ActorComponent> pBase(it->second);
              //std::shared_ptr<ComponentType> pSub();
              std::weak_ptr<ComponentType> pWeakSub(std::tr1::static_pointer_cast<ComponentType>(pBase));
              return pWeakSub;
          }
//          LOG_CRITICAL_ERROR_A("Component not found %d\n", id);
          return std::weak_ptr<ComponentType>();
        }

      template<class ComponentType>
      BOOL HasComponent(ComponentId id)
      {
          return GetComponent<ComponentType>(id).lock() != NULL;
      }
  };
};
