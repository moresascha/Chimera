#pragma once
#include "CMTypes.h"

namespace chimera
{
    class IApplication
    {
    public:
        
        virtual void VClose(void) = 0;

        virtual void VRun(void) = 0;

        virtual IEventManager* VGetEventManager(void) = 0;

        virtual IFileSystem* VGetFileSystem(void) = 0;

        virtual bool VIsRunning(void) = 0;

        virtual void VStopRunning(void) = 0;

        virtual IResourceCache* VGetCache(void) = 0;

        virtual IScript* VGetScript(void) = 0;

        virtual IScriptEventManager* VGetScriptEventManager(void) = 0;

        virtual ILogic* VGetLogic(void) = 0;

        virtual IRenderer* VGetRenderer(void) = 0;

        virtual IHumanView* VGetHumanView(void) = 0;

        virtual ITimer* VGetUpdateTimer(void) = 0;

        virtual ITimer* VGetRenderingTimer(void) = 0;

        virtual IInputHandler* VGetInputHandler(void) = 0;

        virtual IConfig* VGetConfig(void) = 0;

        virtual HWND VGetWindowHandle(void) = 0;

        virtual uint VGetWindowWidth(void) const = 0;

        virtual uint VGetWindowHeight(void) const = 0;
    };

    class IApplicationFactory
    {
    public:
        virtual IApplication* VCreateApplication(void) = 0;
    };

    class ITimer
    {
    public:
        virtual void VTick(void) = 0;

        virtual void VReset(void) = 0;

        virtual float VGetFPS(void) const = 0;

        virtual ulong VGetTime(void) const = 0;

        virtual ulong VGetLastMillis(void) const = 0;

        virtual ulong VGetLastMicros(void) const = 0;
    };

    class IFileSystem
    {
    public:
        virtual void VRegisterFile(LPCSTR name, LPCSTR path) = 0;

        virtual void VRegisterCallback(LPCSTR dllName, LPCSTR path, OnFileChangedCallback cb) = 0;

        virtual void VRemoveCallback(LPCSTR dllName, OnFileChangedCallback cb) = 0;
    };

    class IConfig
    {
    public:
        virtual bool VLoadConfigFile(LPCSTR file) = 0;

        virtual bool VGetBool(LPCSTR value) = 0;

        virtual void VCreateDefaults(void) = 0;

        virtual std::string VGetString(LPCSTR value) = 0;

        virtual int VGetInteger(LPCSTR value) = 0;

        virtual float VGetFloat(LPCSTR value) = 0;

        virtual DOUBLE VGetDouble(LPCSTR value) = 0;

        virtual ~IConfig(void) {}
    };

    class CMIStream
    {
    public:
    };
};