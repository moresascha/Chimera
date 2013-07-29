#include "VRamManager.h"
#include "Geometry.h"
#include "util.h"
#include "Texture.h"
#include <vector>
#include "Mesh.h"
#include "GameApp.h"

namespace tbd 
{

    VRamHandle* GeometryCreator::VGetHandle(VOID)
    {
        return new d3d::Geometry();
    }

    VOID GeometryCreator::VCreateHandle(VRamHandle* handle)
    {
        d3d::Geometry* geo = (d3d::Geometry*)handle;
        std::shared_ptr<tbd::Mesh> mesh = std::static_pointer_cast<tbd::Mesh>(app::g_pApp->GetCache()->GetHandle(handle->GetResource()));
        geo->SetIndexBuffer(mesh->GetIndices(), mesh->GetIndexCount());
        geo->SetVertexBuffer(mesh->GetVertices(), mesh->GetVertexCount(), mesh->GetVertexStride());
    }

    VRamHandle* TextureCreator::VGetHandle(VOID)
    {
        return new d3d::Texture2D();
    }

    VOID TextureCreator::VCreateHandle(VRamHandle* handle)
    {
        d3d::Texture2D* texture = (d3d::Texture2D*)handle;
        std::shared_ptr<tbd::ResHandle> xtraHandle = app::g_pApp->GetCache()->GetHandle(handle->GetResource());
        std::shared_ptr<tbd::ImageExtraData> data = std::static_pointer_cast<tbd::ImageExtraData>(xtraHandle->GetExtraData());

        texture->SetBindflags(D3D11_BIND_SHADER_RESOURCE);
        //texture->GetDescription().BindFlags = D3D11_BIND_SHADER_RESOURCE;

        //texture->GetDescription().Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        texture->SetFormat(DXGI_FORMAT_R8G8B8A8_UNORM);

        //texture->GetDescription().Width = data->m_width;
        texture->SetWidth(data->m_width);

        //texture->GetDescription().Height = data->m_height;
        texture->SetHeight(data->m_height);

        //texture->GetDescription().MipLevels = 0;
        texture->SetMipMapLevels(0);
    
        //texture->GetDescription().SampleDesc.Count = 1;
        texture->SetSamplerCount(1);
    
        //texture->GetDescription().SampleDesc.Quality = 0;
        texture->SetSamplerQuality(0);

        //texture->GetDescription().ArraySize = 1;
        texture->SetArraySize(1);

        //texture->GetDescription().MiscFlags = D3D11_RESOURCE_MISC_GENERATE_MIPS;
        texture->SetMicsFlags(D3D11_RESOURCE_MISC_GENERATE_MIPS);

        texture->SetData(xtraHandle->Buffer());
    }

    //--manager--//
    VRamManager::VRamManager(UINT mb) : m_bytes(1024 * 1024 * mb), m_currentByteSize(0), m_updateFrequency(0.05f) /*every 20 seconds*/, m_time(0) 
    {
        m_creators["obj"] = new GeometryCreator();
        m_creators["png"] = new TextureCreator();
        m_creators["jpg"] = new TextureCreator();
    }

    std::shared_ptr<VRamHandle> VRamManager::GetHandle(CONST VRamResource& ressource)
    {
        return _GetHandle(ressource, FALSE);
    }

    VOID VRamManager::RegisterHandleCreator(LPCSTR suffix, IVRamHandleCreator* creator)
    {
        VRamResource res(suffix);
        if(m_creators.find(res.m_name) != m_creators.end())
        {
            LOG_CRITICAL_ERROR("suffix creator exists already");
        }
        m_creators[res.m_name] = creator;
    }

    
    VOID VRamManager::AppendAndCreateHandle(std::string& name, std::shared_ptr<VRamHandle> handle)
    {
        m_locker.Lock();

        handle->SetResource(name);

        auto it = m_ressources.find(handle->GetResource().m_name);

        if(it != m_ressources.end())
        {
            return;
        }
    
        m_ressources[handle->GetResource().m_name] = handle;

        m_locker.Unlock();

        handle->VCreate();

        handle->SetCreated(TRUE);

        m_currentByteSize += handle->VGetByteCount();

        handle->Update();
    }

    std::shared_ptr<VRamHandle> VRamManager::GetHandleAsync(CONST VRamResource& ressource)
    {
        return _GetHandle(ressource, TRUE);
    }

    std::shared_ptr<VRamHandle> VRamManager::_GetHandle(CONST VRamResource& ressource, BOOL async)
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

        auto it = m_ressources.find(ressource.m_name);

        if(it != m_ressources.end())
        {
            if(it->second->IsReady())
            {
                Update(it->second);
            }

            m_locker.Unlock();
            /*
            if(it->second->IsReady())
            {
               m_locks[ressource.m_name]->Unlock();
            } */
            //DEBUG_OUT("VR returned: " + ressource.m_name);
            return it->second;
        }
   
        //DEBUG_OUT("VR aquir: " + ressource.m_name);

        std::vector<std::string> elems = util::split(ressource.m_name, '.');

        if(elems.size() < 2)
        {
            LOG_CRITICAL_ERROR("Unknown file format");
            return std::shared_ptr<VRamHandle>();
        }

        std::string pattern = elems.back();

        auto cit = m_creators.find(pattern);
        if(cit == m_creators.end())
        {
            LOG_CRITICAL_ERROR_A("no creator installed for pattern: %s", pattern.c_str());
        }

        IVRamHandleCreator* creator = cit->second;

        std::shared_ptr<VRamHandle> handle = std::shared_ptr<VRamHandle>(creator->VGetHandle());

        m_ressources[ressource.m_name] = handle;
    
        handle->m_resource = ressource;

        m_locker.Unlock();

        creator->VCreateHandle(handle.get());

        if(!handle->VCreate())
        {
            LOG_CRITICAL_ERROR_A("failed to create vr ressource: %s", ressource.m_name.c_str());
        }

       // DEBUG_OUT("VR loaded " + ressource.m_name);

        handle->SetCreated(TRUE);
    
        //m_locks[ressource.m_name]->Unlock();

        m_currentByteSize += handle->VGetByteCount();

        handle->Update();

        return handle;
    }

    VOID VRamManager::Update(std::shared_ptr<VRamHandle> ressource)
    {
        ressource->Update();
    }

    VOID VRamManager::Update(ULONG millis)
    {
        m_time += millis;
        FLOAT maxSeconds = 1.0f / m_updateFrequency;

        if(m_time < maxSeconds * 1000.0f)
        {
            return;
        }
        m_time = 0;

        LONG current = clock();
        for(auto it = m_ressources.begin(); it != m_ressources.end();)
        {
            std::shared_ptr<VRamHandle> r = it->second;
            LONG t = current - r->GetLastUsageTime();
           // DEBUG_OUT_A("Lastusage: %s, %d", r->GetResource().m_name.c_str(), t);
            if((t / (FLOAT)CLOCKS_PER_SEC) > maxSeconds && r->IsReady())
            {
                //DEBUG_OUT_A("Deleting: %s, %d", r->GetResource().m_name.c_str(), t);
                it = Free(r);
            }
            else
            {
                ++it;
            }
        }
       // DEBUG_OUT("----");
    }

    std::map<std::string, std::shared_ptr<VRamHandle>>::iterator VRamManager::Free(std::shared_ptr<VRamHandle> ressource)
    {
        auto itt = m_ressources.find(ressource->m_resource.m_name);
        if(itt == m_ressources.end())
        {
            return itt;
        }
        m_currentByteSize -= ressource->VGetByteCount();
        ressource->VDestroy();
        ressource->SetCreated(FALSE);
        delete m_locks[ressource->m_resource.m_name];
        m_locks.erase(ressource->m_resource.m_name);
        auto it = m_ressources.erase(itt);
        return it;
    }

    VOID VRamManager::Flush(VOID)
    {
        for(auto it = m_ressources.begin(); it != m_ressources.end();)
        {
            it = Free(it->second);
        }
        m_ressources.clear();
    }

    VRamManager::~VRamManager(VOID)
    {
        Flush();
        for(auto it = m_creators.begin(); it != m_creators.end(); ++it)
        {
            delete it->second;
        }
        m_creators.clear();
    }
};