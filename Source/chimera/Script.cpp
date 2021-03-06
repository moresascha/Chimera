#include "Script.h"
#include "GameApp.h"
#include "Vec3.h"
#include "LuaHelper.h"
#include "Actor.h"
#include "GameLogic.h"
#include "Components.h"
#include "EventManager.h"
#include "luaplus/LuaPlus.h"
#include "ScriptEvent.h"

#ifdef _DEBUG
    #pragma comment (lib, "LuaPlusD.lib")
#else
    #pragma comment (lib, "LuaPlus.lib")
#endif

namespace chimera
{
    namespace script
    {

#define CHECK_LUA_ERROR(error) CheckError(error)

        bool LuaScript::VInit(void)
        {
            m_pState = LuaPlus::LuaState::Create(true);

            RETURN_IF_FAILED(m_pState);

            ResgisterMemberFunction("RunFile", this, &LuaScript::VRunFile);
            ResgisterMemberFunction("Run", this, &LuaScript::VRunString);

            return true;
        }

        void LuaScript::CheckError(int error)
        {
            if(error != 0)
            {
                LuaPlus::LuaStackObject stackObj(m_pState, -1);
                LPCSTR errstr = stackObj.GetString();
                if(errstr)
                {
                    LOG_ERROR(errstr); //todo
                }
                m_pState->SetTop(0);
            }
        }

        void LuaScript::VRunFile(LPCSTR file)
        {
            CHECK_LUA_ERROR(m_pState->DoFile(file));
        }

        void LuaScript::VRunString(LPCSTR str)
        {
            CHECK_LUA_ERROR(m_pState->DoString(str));
        }

        LuaScript::~LuaScript(void)
        {
            LuaPlus::LuaState::Destroy(m_pState);
        }

        namespace internalexports
        {
            void CreateActor(LPCSTR xmlFile, LuaPlus::LuaObject luaPosition, LuaPlus::LuaObject luaRotation)
            {
                if(!luaPosition.IsTable() || !luaRotation.IsTable())
                {
                    LOG_ERROR_NR("Failed to create Actor 'position' and 'Rotation' have to be a table");
                }
                util::Vec3 position;
                if(!chimera::script::ConvertTableToVec3(position, luaPosition))
                {
                    LOG_ERROR_NR("Failed to create Actor, couldn't parse 'position'");
                }

                util::Vec3 rotation;
                if(!chimera::script::ConvertTableToVec3(rotation, luaRotation))
                {
                    LOG_ERROR_NR("Failed to create Actor, couldn't parse 'rotation'");
                }

                std::shared_ptr<chimera::Actor> actor = chimera::g_pApp->GetLogic()->VCreateActor(xmlFile);
                std::shared_ptr<chimera::MoveActorEvent> move = std::shared_ptr<chimera::MoveActorEvent>(new chimera::MoveActorEvent(actor->GetId(), position, rotation, false));
                chimera::IEventManager::Get()->VQueueEvent(move);
            }

            void MoveActor(LuaPlus::LuaObject luaid, LuaPlus::LuaObject luaPosition, LuaPlus::LuaObject luaRotation, LuaPlus::LuaObject luadelta)
            {
                if(!luaPosition.IsTable() || !luaRotation.IsTable())
                {
                    LOG_ERROR_NR("Failed to MoveActor Actor 'position' and 'Rotation' have to be a table");
                }
                util::Vec3 position;
                if(!chimera::script::ConvertTableToVec3(position, luaPosition))
                {
                    LOG_ERROR_NR("Failed to MoveActor Actor, couldn't parse 'position'");
                }

                util::Vec3 rotation;
                if(!chimera::script::ConvertTableToVec3(rotation, luaRotation))
                {
                    LOG_ERROR_NR("Failed to MoveActor Actor, couldn't parse 'rotation'");
                }

                ActorId id = CM_INVALID_ACTOR_ID;
                if(luaid.IsInteger())
                {
                    id = luaid.GetInteger();
                }

                bool isDelta = false;
                if(luadelta.IsBoolean())
                {
                    isDelta = luadelta.GetBoolean();
                }
                LOG_CRITICAL_ERROR("this doesnt work anymore!");
                //QUEUE_EVENT(new event::MoveActorEvent(id, position, rotation, isDelta));
            }

            void RegisterEventListener(EventType type, LuaPlus::LuaObject funciton)
            {
                chimera::g_pApp->GetScriptEventManager()->AddListener(funciton, type);
            }

            void Print(LuaPlus::LuaObject text)
            {
                if(text.IsBoolean())
                {
                    DEBUG_OUT(text.GetBoolean() ? "true" : "false");
                }
                else if(text.IsNumber())
                {
                    /*if(text.IsInteger())
                    {
                        DEBUG_OUT(text.GetInteger());
                    }
                    else */
                    {
                        DEBUG_OUT_A("%f", text.GetDouble());
                    }
                }
                else if(text.IsString())
                {
                    DEBUG_OUT(text.GetString());
                }
            }
            
            void Register(void)
            {
                LuaScript* script = chimera::g_pApp->GetScript();
                script->ResgisterFunction("CreateActor", &internalexports::CreateActor);
                script->ResgisterFunction("MoveActor", &internalexports::MoveActor);
                script->ResgisterFunction("Printf", &internalexports::Print);
                script->ResgisterFunction("RegisterEventListener", &internalexports::RegisterEventListener);


                chimera::script::RegisterScriptEvents();
            }
        }
    }
}
