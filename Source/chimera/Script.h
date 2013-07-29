#pragma once
#include "stdafx.h"

namespace LuaPlus
{
    class LuaState;
}
namespace tbd
{
    namespace script
    {
        class IScript
        {

        public:
            virtual BOOL VInit(VOID) = 0;
            virtual VOID VRunFile(LPCSTR file) = 0;
            virtual VOID VRunString(LPCSTR str) = 0;
            virtual ~IScript(VOID) {}
        };

        class LuaScript : public IScript
        {
        private:
            LuaPlus::LuaState* m_pState;
            VOID CheckError(INT error);
        public:
            LuaScript(VOID) : m_pState(NULL) {}
            BOOL VInit(VOID);
            VOID VRunFile(LPCSTR file);
            VOID VRunString(LPCSTR str);

            template<typename Caller, typename Func>
            VOID ResgisterMemberFunction(LPCSTR name, Caller caller, Func funciton)
            {
                m_pState->GetGlobals().RegisterDirect(name, *caller, funciton);
            }

            template<typename Func>
            VOID ResgisterFunction(LPCSTR name, Func funciton)
            {
                m_pState->GetGlobals().RegisterDirect(name, funciton);
            }

            LuaPlus::LuaState* GetState(VOID)
            {
                return m_pState;
            }

            ~LuaScript(VOID);
        };

        namespace internalexports
        {
            VOID Register(VOID);    
        }
    }
}


