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
        uint m_bytes;
        uint m_currentByteSize;
        float m_updateFrequency;
        ulong m_time;
        util::Locker m_locker;
        std::map<std::string, std::shared_ptr<IVRamHandle>> m_resources;
        std::map<std::string, IVRamHandleCreator*> m_creators;
        std::map<std::string, util::Locker*> m_locks;

        std::shared_ptr<IVRamHandle> _GetHandle(const VRamResource& ressource, bool async);

        void Reload(std::shared_ptr<IVRamHandle> handle);

        std::map<std::string, std::shared_ptr<IVRamHandle>>::iterator Free(std::shared_ptr<IVRamHandle> ressource);

    public:
        VRamManager(uint mb);

        uint VGetCurrentSize(void) const { return m_currentByteSize; }

        uint VGetMaxSize(void) const { return m_bytes; }

        float VGetWorkload(void) const { return 100.0f * (float)m_currentByteSize / (float)m_bytes; }

        void VUpdate(ulong millis);

        void VFlush(void);

        void VFree(std::shared_ptr<IVRamHandle> ressource) { LOG_CRITICAL_ERROR("TODO");}

        std::shared_ptr<IVRamHandle> VGetHandle(const VRamResource& ressource);

        void VRegisterHandleCreator(LPCSTR suffix, IVRamHandleCreator* creator);

        std::shared_ptr<IVRamHandle> VGetHandleAsync(const VRamResource& ressource);

        void VAppendAndCreateHandle(std::shared_ptr<IVRamHandle> handle);

        void OnResourceChanged(std::shared_ptr<IEvent> event);

        ~VRamManager(void);
    };
}