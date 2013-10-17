#include "ScriptEvent.h"
#include "GameApp.h"
#include "Event.h"
#include "EventManager.h"
namespace chimera
{
    namespace script
    {
        LuaScriptEventTransformation* LuaScriptEventManager::GetTransformation(EventType type)
        {
            auto it = m_transformMap.find(type);

            if(it == m_transformMap.end())
            {
                return NULL;
            }
            else
            {
                return &it->second;
            }
        }

        VOID LuaScriptEventManager::RegisterEventTransformer(LuaScriptEventTransformation transformer, EventType type, LPCSTR name)
        {
            LuaScript* luas = (LuaScript*)chimera::g_pApp->GetScript();
            LuaPlus::LuaObject table = luas->GetState()->GetGlobals().GetByName("EventType");

            if(table.IsNil())
            {
                table = luas->GetState()->GetGlobals().CreateTable("EventType");
            }

            table.SetNumber(name, type);

            m_transformMap[type] = transformer;
        }

        VOID LuaScriptEventManager::AddListener(LuaPlus::LuaObject listener, EventType type)
        {
            LuaEventListener* l = new LuaEventListener(listener);
            m_listenerMap[type].push_back(std::shared_ptr<LuaEventListener>(l));

            chimera::EventListener elistener = fastdelegate::MakeDelegate(l, &LuaEventListener::EventListenerDelegate);
            chimera::IEventManager::Get()->VAddEventListener(elistener, type);
        }

        VOID LuaScriptEventManager::RemoveListener(LuaPlus::LuaObject listener, EventType type)
        {
            auto it = m_listenerMap.find(type);
            for(auto itt = it->second.begin(); itt != it->second.end(); ++itt)
            {
                LuaEventListener* l = itt->get();
                if(listener.GetString() == l->GetFunction().GetString())
                {
                    itt = it->second.erase(itt);
                    chimera::EventListener elistener = fastdelegate::MakeDelegate(l, &LuaEventListener::EventListenerDelegate);
                    chimera::IEventManager::Get()->VRemoveEventListener(elistener, type);
                }
            }
        }

        VOID LuaEventListener::EventListenerDelegate(event::IEventPtr event)
        {
            LuaScriptEventTransformation* t = chimera::g_pApp->GetScriptEventManager()->GetTransformation(event->VGetEventType());
            if(t)
            {
                if(!m_function.IsFunction())
                {
                    LOG_ERROR_NR((std::string(m_function.GetString()) + std::string(" is not a function")).c_str());
                }
                LuaPlus::LuaFunction<VOID> function(m_function);
                
                LuaPlus::LuaObject obj = t->VBuildForScript(event);
                if(obj.IsNil())
                {
                    function();
                }
                else
                {
                    function(obj);
                }
            }
            else
            {
                LOG_ERROR("no Transformation");
            }
        }

        VOID RegisterScriptEvents(VOID)
        {
            REGISTER_SCRIPT_TRANSFORMER(LuaScriptEventTransformation(), chimera::ActorCreatedEvent::TYPE, ActorCreatedEvent);
        }
    }
}
