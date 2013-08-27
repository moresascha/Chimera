#pragma once
#include "stdafx.h"
#include <time.h>
#include "Resources.h"
#include "Locker.h"

namespace event
{
    class IEvent;
}
namespace tbd 
{

    typedef tbd::Resource VRamResource;

    class VRamHandle 
    {
        friend class VRamManager;
        friend class IVRamHandleCreator;
    protected:
        LONG m_lastUsage;
        BOOL m_created;
        VRamResource m_resource;
        //std::shared_ptr<tbd::ResHandle> m_handle;
        VOID SetResource(std::string& name);
    public:
        VRamHandle(VOID);
        virtual BOOL VCreate(VOID) = 0;
        virtual VOID VDestroy() = 0;
        virtual UINT VGetByteCount(VOID) CONST = 0;

        VOID SetCreated(BOOL created) { m_created = created; };
        BOOL IsReady(VOID) CONST;
        VOID Update(VOID) { m_lastUsage = clock(); }
        LONG GetLastUsageTime() CONST { return m_lastUsage; }
        VRamResource& GetResource(VOID) { return m_resource; }
    };

    class IVRamHandleCreator
    {
    public:
        virtual VRamHandle* VGetHandle(VOID) = 0;
        virtual VOID VCreateHandle(VRamHandle* handle) = 0;
    };

    class GeometryCreator : public IVRamHandleCreator
    {
    public:
        VRamHandle* VGetHandle(VOID);
        VOID VCreateHandle(VRamHandle* handle);
    };

    class TextureCreator : public IVRamHandleCreator
    {
    public:
        VRamHandle* VGetHandle(VOID);
        VOID VCreateHandle(VRamHandle* handle);
    };

    class VRamManager
    {
    private:
        UINT m_bytes;
        UINT m_currentByteSize;
        FLOAT m_updateFrequency;
        ULONG m_time;
        util::Locker m_locker;
        std::map<std::string, std::shared_ptr<VRamHandle>> m_resources;
        std::map<std::string, IVRamHandleCreator*> m_creators;
        std::map<std::string, util::Locker*> m_locks;

        std::shared_ptr<VRamHandle> _GetHandle(CONST VRamResource& ressource, BOOL async);
        VOID Update(std::shared_ptr<VRamHandle> ressource);

        VOID Reload(std::shared_ptr<VRamHandle> handle);

    public:
        VRamManager(UINT mb);

        UINT GetCurrentSize(VOID) CONST { return m_currentByteSize; }

        UINT GetMaxSize(VOID) CONST { return m_bytes; }

        FLOAT GetWorkload(VOID) CONST { return 100.0f * (FLOAT)m_currentByteSize / (FLOAT)m_bytes; }

        VOID Update(ULONG millis);

        VOID Flush(VOID);

        std::map<std::string, std::shared_ptr<VRamHandle>>::iterator Free(std::shared_ptr<VRamHandle> ressource);

        std::shared_ptr<VRamHandle> GetHandle(CONST VRamResource& ressource);

        VOID RegisterHandleCreator(LPCSTR suffix, IVRamHandleCreator* creator);

        std::shared_ptr<VRamHandle> GetHandleAsync(CONST VRamResource& ressource);

        VOID OnResourceChanged(std::shared_ptr<event::IEvent> event);

        VOID AppendAndCreateHandle(std::string& name, std::shared_ptr<VRamHandle> handle);

        ~VRamManager(VOID);
    };
}