#pragma once
#include "stdafx.h"

namespace chimera 
{
    extern "C" LRESULT CALLBACK WndProc(HWND, uint, WPARAM, LPARAM);

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
        bool m_running;
        ulong m_updateInterval; //freq per second
        DOUBLE m_updateFreqMillis;
        IFileSystem* m_pFileSystem;

        static HINSTANCE g_hInstance;

        Application(void);

        virtual bool VInitialise(FactoryPtr* facts);
        
        void VClose(void);

        void Update(void);

        void VRun(void);

        void Render(void);

        bool VIsRunning(void) { return m_running; }

        void VStopRunning(void) { m_running = false; }

        IHumanView* VGetHumanView(void);

        IEventManager* VGetEventManager(void) { return m_pEventMgr; }

        IFileSystem* VGetFileSystem(void) { return m_pFileSystem; }

        IResourceCache* VGetCache(void) { return m_pCache; }

        IScript* VGetScript(void) { return m_pScript; }

        IScriptEventManager* VGetScriptEventManager(void) { return m_pScriptEventManager; }

        ILogic* VGetLogic(void) { return m_pLogic; }

        IRenderer* VGetRenderer(void) { return VGetHumanView()->VGetRenderer(); } //we keep it here to show loading modules, todo: move to gameview

        ITimer* VGetUpdateTimer(void) { return m_pUpdateTimer; }

        ITimer* VGetRenderingTimer(void) { return m_pRenderingTimer; }

        IInputHandler* VGetInputHandler(void) { return m_pInput; }

        IConfig* VGetConfig(void) { return m_pConfig; }

        HWND VGetWindowHandle(void);

        uint VGetWindowWidth(void) const;

        uint VGetWindowHeight(void) const;

        bool VIsRunning(void) const;

        virtual ~Application(void);
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

