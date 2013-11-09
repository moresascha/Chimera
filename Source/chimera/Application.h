#pragma once
#include "stdafx.h"

namespace chimera 
{
    extern "C" LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

    class Application : public IApplication
    {
    public:
        ILogic* m_pLogic;
        IInputHandler* m_pInput;
        ITimer* m_timer0;
        ITimer* m_pUpdateTimer;
        ITimer* m_pRenderingTimer;
        IResourceCache* m_pCache;
        IEventManager* m_pEventMgr;
        IScript* m_pScript;
        IScriptEventManager* m_pScriptEventManager;
        IConfig* m_pConfig;
        BOOL m_running;
        ULONG m_updateInterval; //freq per second
        DOUBLE m_updateFreqMillis;
        IFileSystem* m_pFileSystem;

        static HINSTANCE g_hInstance;

        Application(VOID);

        virtual BOOL VInitialise(FactoryPtr* facts);
        
        VOID VClose(VOID);

        VOID Update(VOID);

        VOID VRun(VOID);

        VOID Render(VOID);

        BOOL VIsRunning(VOID) { return m_running; }

        VOID VStopRunning(VOID) { m_running = FALSE; }

        IHumanView* VGetHumanView(VOID);

        IEventManager* VGetEventManager(VOID) { return m_pEventMgr; }

        IFileSystem* VGetFileSystem(VOID) { return m_pFileSystem; }

        IResourceCache* VGetCache(VOID) { return m_pCache; }

        IScript* VGetScript(VOID) { return m_pScript; }

        IScriptEventManager* VGetScriptEventManager(VOID) { return m_pScriptEventManager; }

        ILogic* VGetLogic(VOID) { return m_pLogic; }

        IRenderer* VGetRenderer(VOID) { return VGetHumanView()->VGetRenderer(); } //we keep it here to show loading modules, todo: move to gameview

        ITimer* VGetUpdateTimer(VOID) { return m_pUpdateTimer; }

        ITimer* VGetRenderingTimer(VOID) { return m_pRenderingTimer; }

        IInputHandler* VGetInputHandler(VOID) { return m_pInput; }

        IConfig* VGetConfig(VOID) { return m_pConfig; }

        HWND VGetWindowHandle(VOID);

        UINT VGetWindowWidth(VOID) CONST;

        UINT VGetWindowHeight(VOID) CONST;

        BOOL VIsRunning(VOID) CONST;

        virtual ~Application(VOID);
    };

    extern Application* g_pApp;
}

/*

BOOL IsRunning(VOID) { return m_running; }

VOID End(VOID) { m_running = FALSE; }

virtual VOID VCreateLogicAndView(VOID) = 0;

chimera::IEventManager* VGetEventMgr(VOID) CONST { return m_pEventMgr; }

chimera::IHumanGameView* VGetHumanView(VOID) CONST { return m_pHumanView; }

chimera::IFileSystem* VGetFileSystem(VOID) CONST { return m_pFileSystem; }

chimera::IResourceCache* VGetCache(VOID) { return m_pCache; }

chimera::script::IScript* VGetScript(VOID) { return m_pScript; }

chimera::script::LuaScriptEventManager* GetScriptEventManager(VOID) { return m_pScriptEventManager; }

chimera::IGameLogic* VGetLogic(VOID) CONST { return m_pLogic; }

chimera::IRenderer* VGetRenderer(VOID) CONST { return m_pRenderer; } //we keep it here to show loading modules, todo: move to gameview

chimera::IVRamManager* GetVRamManager(VOID) { return m_pVramManager; }

util::ITimer* VGetUpdateTimer(VOID) { return &m_updateTimer; }

util::ITimer* VGetRenderingTimer(VOID) { return &m_renderingTimer; }

chimera::IInputHandler* VGetInputHandler(VOID) { return m_pInput; }

chimera::IFontManager* VGetFontManager(VOID) { return m_pFontManager; } //we keep it here to show loading modules, todo: move to gameview

chimera::IConfig* VGetConfig(VOID) CONST { return m_pConfig; }
*/

