#pragma once
#include "stdafx.h"
#include "Timer.h"
#include "Config.h"

namespace tbd
{
    class InputHandler;
    class HumanGameView;
    class ResourceCache;
    class FontManager;
    class IFontRenderer;
    class VRamManager;
    class BaseGameLogic;
}

namespace tbd
{
    namespace script
    {
        class IScript;
        class LuaScript;
        class LuaScriptEventManager;
    }
}

namespace d3d
{
    class D3DRenderer;
}

namespace event
{
    class EventManager;
}

namespace app 
{
    VOID PostInitMessage(LPCSTR message);

    class GameApp
    {
    protected:
        tbd::BaseGameLogic* m_pLogic;
        tbd::InputHandler* m_pInput;
        util::Timer m_timer0;
        util::Timer m_updateTimer;
        util::Timer m_renderingTimer;
        tbd::ResourceCache* m_pCache;
        event::EventManager* m_pEventMgr;
        tbd::script::LuaScript* m_pScript;
        tbd::script::LuaScriptEventManager* m_pScriptEventManager;
        tbd::HumanGameView* m_pHumanView;
        tbd::VRamManager* m_pVramManager;
        d3d::D3DRenderer* m_pRenderer;
        tbd::FontManager* m_pFontManager;
        tbd::IFontRenderer* m_pFontRenderer;
        util::Config m_config;
        BOOL m_running;
        ULONG m_updateInterval; //freq per second
        DOUBLE m_updateFreqMillis;

    public:

        static HINSTANCE g_hInstance;

        GameApp(VOID);

        BOOL Init(CONST HINSTANCE hInstance, LPCWSTR title, std::string resCache, std::string logFile, ULONG updateIval = 60);

        VOID Update(VOID);

        VOID Run(VOID);

        VOID Render(VOID);

        BOOL IsRunning(VOID) { return m_running; }

        VOID End(VOID) { m_running = FALSE; }

        virtual VOID VCreateLogicAndView(VOID) = 0;

        event::EventManager* GetEventMgr(VOID) CONST { return m_pEventMgr; }

        tbd::HumanGameView* GetHumanView(VOID) CONST { return m_pHumanView; }

        tbd::ResourceCache* GetCache(VOID) { return m_pCache; }

        tbd::script::LuaScript* GetScript(VOID) { return m_pScript; }

        tbd::script::LuaScriptEventManager* GetScriptEventManager(VOID) { return m_pScriptEventManager; }

        tbd::BaseGameLogic* GetLogic(VOID) { return m_pLogic; }

        d3d::D3DRenderer* GetRenderer(VOID) { return m_pRenderer; } //we keep it here to show loading modules, todo: move to gameview

        tbd::VRamManager* GetVRamManager(VOID) { return m_pVramManager; }

        util::Timer* GetUpdateTimer(VOID) { return &m_updateTimer; }

        util::Timer* GetRenderingTimer(VOID) { return &m_renderingTimer; }

        tbd::InputHandler* GetInputHandler(VOID) { return m_pInput; }

        tbd::FontManager* GetFontManager(VOID) { return m_pFontManager; } //we keep it here to show loading modules, todo: move to gameview

        util::Config* GetConfig(VOID) { return &m_config; }

        HWND GetWindowHandle(VOID);

        UINT GetWindowWidth(VOID) CONST;

        UINT GetWindowHeight(VOID) CONST;

        virtual ~GameApp(VOID);
    };

    extern GameApp* g_pApp;
}

