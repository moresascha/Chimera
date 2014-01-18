#include "VRamManager.h"
#include "util.h"
#include "Mesh.h"
#include "Event.h"

namespace chimera 
{
    //--manager--//
    VRamManager::VRamManager(uint mb) : m_bytes(1024 * 1024 * mb), m_currentByteSize(0), m_updateFrequency(0.05f) /*every 20 seconds*/, m_time(0) 
    {
    }

    void VRamManager::OnResourceChanged(std::shared_ptr<IEvent> data)
    {
        std::shared_ptr<ResourceChangedEvent> event = std::static_pointer_cast<ResourceChangedEvent>(data);
        auto h = m_resources.find(event->m_resource);//GetHandle(tbd::VRamResource(event->m_resource));

        if(h != m_resources.end())
        {
            Reload(h->second);
        }
    }

    void VRamManager::Reload(std::shared_ptr<IVRamHandle> handle)
    {
        handle->VSetReady(false);
    }

    std::shared_ptr<IVRamHandle> VRamManager::VGetHandle(const VRamResource& ressource)
    {
        return _GetHandle(ressource, false);
    }

    void VRamManager::VRegisterHandleCreator(LPCSTR suffix, IVRamHandleCreator* creator)
    {
        VRamResource res(suffix);
        if(m_creators.find(res.m_name) != m_creators.end())
        {
            LOG_CRITICAL_ERROR("suffix creator exists already");
        }
        m_creators[res.m_name] = creator;
    }

    
    void VRamManager::VAppendAndCreateHandle(std::shared_ptr<IVRamHandle> handle)
    {
        m_locker.Lock();

        auto it = m_resources.find(handle->VGetResource().m_name);

        if(it != m_resources.end())
        {
            LOG_CRITICAL_ERROR_A("A handle with the resource '%s' already exists!", handle->VGetResource().m_name.c_str());
        }
    
        m_resources[handle->VGetResource().m_name] = handle;

        m_locker.Unlock();

        handle->VCreate();

        handle->VSetReady(true);

        m_currentByteSize += handle->VGetByteCount();

        handle->VUpdate();
    }

    std::shared_ptr<IVRamHandle> VRamManager::VGetHandleAsync(const VRamResource& ressource)
    {
        return _GetHandle(ressource, true);
    }

    std::shared_ptr<IVRamHandle> VRamManager::_GetHandle(const VRamResource& ressource, bool async)
    {

        /*
        auto itt = m_locks.find(ressource.m_name);

        if(itt == m_locks.end())
        {
            m_locks[ressource.m_name] = new util::Locker();
        } */
    
       // m_locker.Unlock();
        /*
        m_locks[ressource.m_name]->Lock();

        m_locker.Lock(); */

        m_locker.Lock();

        auto it = m_resources.find(ressource.m_name);

        if(it != m_resources.end())
        {
            //if(it->second->VIsReady())
            {
                it->second->VUpdate();
                m_locker.Unlock();
                return it->second;
            }
            //Free(it->second);
            //return it->second;
        }
   
        //DEBUG_OUT("VR aquir: " + ressource.m_name);

        std::vector<std::string> elems = util::split(ressource.m_name, '.');

        if(elems.size() < 2)
        {
            LOG_CRITICAL_ERROR("Unknown file format");
            return std::shared_ptr<IVRamHandle>();
        }

        std::string pattern = elems.back();

        auto cit = m_creators.find(pattern);
        if(cit == m_creators.end())
        {
            LOG_CRITICAL_ERROR_A("no creator installed for pattern: %s", pattern.c_str());
        }

        IVRamHandleCreator* creator = cit->second;

        std::shared_ptr<IVRamHandle> handle = std::shared_ptr<IVRamHandle>(creator->VGetHandle());

        m_resources[ressource.m_name] = handle;
    
        handle->VSetResource(ressource);

        m_locker.Unlock();

        creator->VCreateHandle(handle.get());

        if(!handle->VCreate())
        {
            LOG_CRITICAL_ERROR_A("failed to create vr ressource: %s", ressource.m_name.c_str());
        }

       // DEBUG_OUT("VR loaded " + ressource.m_name);

        handle->VSetReady(true);
    
        //m_locks[ressource.m_name]->Unlock();

        m_currentByteSize += handle->VGetByteCount();

        handle->VUpdate();

        return handle;
    }

    void VRamManager::VUpdate(ulong millis)
    {
        m_time += millis;
        float maxSeconds = 1.0f / m_updateFrequency;

        if(m_time < maxSeconds * 1000.0f)
        {
            return;
        }
        m_time = 0;

        LONG current = clock();
        for(auto it = m_resources.begin(); it != m_resources.end();)
        {
            std::shared_ptr<IVRamHandle>& r = it->second;
            LONG t = current - r->VGetLastUsageTime();
            if((t / (float)CLOCKS_PER_SEC) > maxSeconds && r->VIsReady())
            {
                it = Free(r);
            }
            else
            {
                ++it;
            }
        }
    }

    std::map<std::string, std::shared_ptr<IVRamHandle>>::iterator VRamManager::Free(std::shared_ptr<IVRamHandle> ressource)
    {
        auto itt = m_resources.find(ressource->VGetResource().m_name);
        if(itt == m_resources.end())
        {
            return itt;
        }
        m_currentByteSize -= ressource->VGetByteCount();
        ressource->VDestroy();
        ressource->VSetReady(false);
        delete m_locks[ressource->VGetResource().m_name];
        m_locks.erase(ressource->VGetResource().m_name);
        auto it = m_resources.erase(itt);
        return it;
    }

    void VRamManager::VFlush(void)
    {
        for(auto it = m_resources.begin(); it != m_resources.end();)
        {
            it = Free(it->second);
        }
        m_resources.clear();
    }

    VRamManager::~VRamManager(void)
    {
        VFlush();
        for(auto it = m_creators.begin(); it != m_creators.end(); ++it)
        {
            delete it->second;
        }
        m_creators.clear();
    }
};