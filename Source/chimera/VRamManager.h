#pragma once
#include "stdafx.h"
#include <time.h>
#include "Locker.h"

namespace chimera
{
    class IEvent;
}
namespace chimera 
{
    class VRamManager : public IVRamManager
    {
    private:
        UINT m_bytes;
        UINT m_currentByteSize;
        FLOAT m_updateFrequency;
        ULONG m_time;
        util::Locker m_locker;
        std::map<std::string, std::shared_ptr<IVRamHandle>> m_resources;
        std::map<std::string, IVRamHandleCreator*> m_creators;
        std::map<std::string, util::Locker*> m_locks;

        std::shared_ptr<IVRamHandle> _GetHandle(CONST VRamResource& ressource, BOOL async);

        VOID Reload(std::shared_ptr<IVRamHandle> handle);

        std::map<std::string, std::shared_ptr<IVRamHandle>>::iterator Free(std::shared_ptr<IVRamHandle> ressource);

    public:
        VRamManager(UINT mb);

        UINT VGetCurrentSize(VOID) CONST { return m_currentByteSize; }

        UINT VGetMaxSize(VOID) CONST { return m_bytes; }

        FLOAT VGetWorkload(VOID) CONST { return 100.0f * (FLOAT)m_currentByteSize / (FLOAT)m_bytes; }

        VOID VUpdate(ULONG millis);

        VOID VFlush(VOID);

        VOID VFree(std::shared_ptr<IVRamHandle> ressource) { LOG_CRITICAL_ERROR("TODO");}

        std::shared_ptr<IVRamHandle> VGetHandle(CONST VRamResource& ressource);

        VOID VRegisterHandleCreator(LPCSTR suffix, IVRamHandleCreator* creator);

        std::shared_ptr<IVRamHandle> VGetHandleAsync(CONST VRamResource& ressource);

        VOID VAppendAndCreateHandle(std::shared_ptr<IVRamHandle> handle);

        VOID OnResourceChanged(std::shared_ptr<IEvent> event);

        ~VRamManager(VOID);
    };
}