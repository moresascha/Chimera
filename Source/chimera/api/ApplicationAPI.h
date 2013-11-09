#pragma once
#include "CMTypes.h"
namespace chimera
{
    class IApplication
    {
    public:
        
        virtual VOID VClose(VOID) = 0;

        virtual VOID VRun(VOID) = 0;

        virtual IEventManager* VGetEventManager(VOID) = 0;

        virtual IFileSystem* VGetFileSystem(VOID) = 0;

        virtual BOOL VIsRunning(VOID) = 0;

        virtual VOID VStopRunning(VOID) = 0;

        virtual IResourceCache* VGetCache(VOID) = 0;

        virtual IScript* VGetScript(VOID) = 0;

        virtual IScriptEventManager* VGetScriptEventManager(VOID) = 0;

        virtual ILogic* VGetLogic(VOID) = 0;

        virtual IRenderer* VGetRenderer(VOID) = 0;

        virtual IHumanView* VGetHumanView(VOID) = 0;

        virtual ITimer* VGetUpdateTimer(VOID) = 0;

        virtual ITimer* VGetRenderingTimer(VOID) = 0;

        virtual IInputHandler* VGetInputHandler(VOID) = 0;

        virtual IConfig* VGetConfig(VOID) = 0;

        virtual HWND VGetWindowHandle(VOID) = 0;

        virtual UINT VGetWindowWidth(VOID) CONST = 0;

        virtual UINT VGetWindowHeight(VOID) CONST = 0;
    };

    class IApplicationFactory
    {
    public:
        virtual IApplication* VCreateApplication(VOID) = 0;
    };

    class ITimer
    {
    public:
        virtual VOID VTick(VOID) = 0;

        virtual VOID VReset(VOID) = 0;

        virtual FLOAT VGetFPS(VOID) CONST = 0;

        virtual ULONG VGetTime(VOID) CONST = 0;

        virtual ULONG VGetLastMillis(VOID) CONST = 0;

        virtual ULONG VGetLastMicros(VOID) CONST = 0;
    };

    class IFileSystem
    {
    public:
        virtual VOID VRegisterFile(LPCSTR name, LPCSTR path) = 0;

        virtual VOID VRegisterCallback(LPCSTR dllName, LPCSTR path, OnFileChangedCallback cb) = 0;

        virtual VOID VRemoveCallback(LPCSTR dllName, OnFileChangedCallback cb) = 0;
    };

    class IConfig
    {
    public:
        virtual BOOL VLoadConfigFile(LPCSTR file) = 0;

        virtual BOOL VGetBool(LPCSTR value) = 0;

        virtual std::string VGetString(LPCSTR value) = 0;

        virtual INT VGetInteger(LPCSTR value) = 0;

        virtual FLOAT VGetFloat(LPCSTR value) = 0;

        virtual DOUBLE VGetDouble(LPCSTR value) = 0;

        virtual ~IConfig(VOID) {}
    };

    class CMIStream
    {
    public:
    };
};