#pragma once
#include "stdafx.h"
#include "Script.h"
#include "Event.h"
#include "luaplus/LuaPlus.h"

#define REGISTER_SCRIPT_TRANSFORMER(transformer, type, classs) \
    app::g_pApp->GetScriptEventManager()->RegisterEventTransformer(transformer, type, #classs) \

namespace tbd
{
    namespace script
    {
        class LuaScriptEventTransformation
        {
            friend class LuaScriptEventManager;
        public:
            virtual LuaPlus::LuaObject VBuildForScript(event::IEventPtr event) { return LuaPlus::LuaObject(); }
            virtual event::IEventPtr VBuildFromScript(LuaPlus::LuaObject) { return NULL; };
        };

        class LuaEventListener
        {
        private:
            LuaPlus::LuaObject m_function;
        public:
            LuaEventListener(LuaPlus::LuaObject function) : m_function(function)
            {

            }
            LuaPlus::LuaObject GetFunction(VOID) { return m_function; }
            VOID EventListenerDelegate(event::IEventPtr event);
        };

        class LuaScriptEventManager
        {
        private:
            std::map<EventType, LuaScriptEventTransformation> m_transformMap;
            std::map<EventType, std::list<std::shared_ptr<LuaEventListener>>> m_listenerMap;
        public:
            VOID AddListener(LuaPlus::LuaObject listener, EventType type);
            VOID RemoveListener(LuaPlus::LuaObject listener, EventType type);
            LuaScriptEventTransformation* GetTransformation(EventType type);
            VOID RegisterEventTransformer(LuaScriptEventTransformation transformer, EventType type, LPCSTR name);
        };

        VOID RegisterScriptEvents(VOID);
    }
}

