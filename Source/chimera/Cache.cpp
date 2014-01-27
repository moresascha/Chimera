#include "Cache.h"
#include "util.h"
#include <fstream>
#include "Mesh.h"
#include "Material.h"
#include "Process.h"
#include <sys/stat.h>
#include <sys/types.h>

namespace chimera 
{
    class WatchResourceModification : public WatchFileModificationProcess
    {
        class ResHandle;
    private:
        std::string m_resource;
    public:
        WatchResourceModification(std::string res, LPCTSTR file, LPCTSTR folder) : WatchFileModificationProcess(file, folder), m_resource(res)
        {
        }

        void VOnFileModification(void)
        {
            QUEUE_EVENT_TSAVE(new chimera::ResourceChangedEvent(m_resource));
            DEBUG_OUT_A("change on Res: %s\n", m_resource.c_str());
        }
    };

    class LoadResourceAsync : public RealtimeProcess
    {
    private:
        IResourceCache* m_pCache;
        CMResource m_res;
        OnResourceLoadedCallback m_cb;

    public:
        LoadResourceAsync(IResourceCache* cache, CMResource& res, OnResourceLoadedCallback cb) : m_pCache(cache), m_res(res), m_cb(cb)
        {
            SetPriority(THREAD_PRIORITY_HIGHEST);
        }

        void VThreadProc(void)
        {
            std::shared_ptr<IResHandle> handle = m_pCache->VGetHandle(m_res);
            if(m_cb)
            {
                m_cb(handle);
            }
            Succeed();
        }
    };

    class LoadFunctionHolder 
    {
    public:
        static void Load(CMResource& r, ResourceCache* cache, IResourceFile* pFile, IResourceLoader* loader, IResourceDecompressor* decomp, std::shared_ptr<IResHandle> handle)
        {
            char* buffer = NULL;
            chimera::CMResource tmp(loader->VSubFolder() + r.m_name);
            uint size = pFile->VGetRawRessource(tmp, &buffer);

            if(!pFile->VIsCompressed())
            {
                size = loader->VLoadRessource(buffer, size, handle);
                if(!size)
                {
                    LOG_CRITICAL_ERROR_A("failed to load ressource: %s", r.m_name.c_str());
                }
            }
            else
            {
                char* dst = NULL;
                uint newSize = decomp->VDecompressRessource(buffer, size, &dst);
                size = loader->VLoadRessource(dst, newSize, handle);
                if(!size)
                {
                    LOG_CRITICAL_ERROR_A("failed to load ressource: %s", r.m_name.c_str());
                }
                if(buffer)
                {
                    delete[] buffer; //old
                }
            }

            handle->VSetReady(true);
            handle->VSetSize(size);

            cache->AllocSilent(size);

#ifdef _DEBUG
            if(CmGetApp()->VGetLogic())
            {
                std::string f = cache->VGetFile().VGetName();
                
                if(loader->VSubFolder() != "")
                {
                     f += loader->VSubFolder();
                }

                std::vector<std::string> split;
                util::split(handle->VGetResource().m_name, '/', split);
                for(int i = 0; i < split.size()-1; ++i)
                {
                    f += split[i] + "/";
                }
                std::string rawFileName = split.back();

                std::wstring wf(f.begin(), f.end());
                std::wstring rwf(rawFileName.begin(), rawFileName.end());
                CmGetApp()->VGetLogic()->VGetProcessManager()->VAttach(
                    std::shared_ptr<WatchResourceModification>(new WatchResourceModification(handle->VGetResource().m_name, rwf.c_str(), wf.c_str())));
            }
#endif
        }
    };

    ResourceCache::ResourceCache(void) : m_defaultDecompressor(NULL), m_defaultLoader(NULL), m_currentCacheSize(NULL)
    {
        ADD_EVENT_LISTENER(this, &ResourceCache::OnResourceChanged, CM_EVENT_RESOURCE_CHANGED);
    }

    void ResourceCache::OnResourceChanged(IEventPtr data)
    {
        std::shared_ptr<ResourceChangedEvent> event = std::static_pointer_cast<ResourceChangedEvent>(data);
        std::shared_ptr<IResHandle> h = Find(CMResource(event->m_resource));
        if(h)
        {
            h->VSetReady(false);
        }
    }

    bool ResourceCache::VInit(uint mbSize, IResourceFile* resFile) 
    {
         m_maxCacheSize = 1024 * 1024 * mbSize;
         m_pFile = resFile; 
        if(m_pFile->VOpen())
        {
            m_defaultDecompressor = new DefaultRessourceDecompressor();
            m_defaultLoader = new DefaultRessourceLoader();
            return true;
        }
        return false;
    }

    void ResourceCache::VRegisterLoader(std::unique_ptr<IResourceLoader> loader) 
    {
        m_loader[loader->VGetPattern()] = std::move(loader);
    }

    void ResourceCache::VRegisterDecompressor(std::unique_ptr<IResourceDecompressor> decompressor) 
    {
        m_decompressor[decompressor->VGetPattern()] = std::move(decompressor);
    }

    void ResourceCache::VAppendHandle(std::unique_ptr<IResHandle> handle)
    {
        m_lock.Lock();
        const CMResource& r = handle->VGetResource();
        auto it = m_locks.find(r.m_name);
        if(it == m_locks.end())
        {
            m_locks[r.m_name] = new util::Locker();

            std::shared_ptr<IResHandle> shrdHanel(std::move(handle));
            m_handlesList.push_front(shrdHanel);

            m_handlesmap[r.m_name] = shrdHanel;

            shrdHanel->VSetResourceCache(this);
            shrdHanel->VSetReady(true);
            AllocSilent(shrdHanel->VSize());
        }
        m_lock.Unlock();
    }

    void ResourceCache::VGetHandleAsync(CMResource& r, OnResourceLoadedCallback cb) 
    {
        std::shared_ptr<IProcess> proc = std::shared_ptr<IProcess>(new LoadResourceAsync(this, r, cb));
        CmGetApp()->VGetLogic()->VGetProcessManager()->VAttach(proc);
    }

    std::shared_ptr<IResHandle> ResourceCache::VGetHandle(CMResource& r) 
    {
        m_lock.Lock();
    
        auto it = m_locks.find(r.m_name.c_str());

        if(it == m_locks.end())
        {
            m_locks[r.m_name] = new util::Locker();
        }

        m_lock.Unlock();

        m_locks[r.m_name]->Lock();

        m_lock.Lock();

        std::shared_ptr<IResHandle> handle = Find(r);

        if(handle) 
        {
            if(handle->VIsReady())
            {
                m_locks[r.m_name]->Unlock();
                Update(handle);
                m_lock.Unlock();
                return handle;
            }
            m_handlesmap.erase(handle->VGetResource().m_name);
            m_handlesList.remove(handle);
            char* d = handle->VBuffer();
            SAFE_ARRAY_DELETE(d);
            MemoryHasBeenFreed(handle->VSize());
        }

        handle = Load(r);

        return handle;
    }

    std::shared_ptr<IResHandle> ResourceCache::Find(CMResource& r) 
    {
        auto handle = m_handlesmap.find(r.m_name);
        if(handle != m_handlesmap.end()) 
        {
            return handle->second;
        }
        return NULL;
    }

    bool ResourceCache::VIsLoaded(CMResource& r)
    {
        std::shared_ptr<IResHandle> handle = Find(r);
        if(handle)
        {
            return handle->VIsReady();
        }
        return false;
    }

    std::shared_ptr<IResHandle> ResourceCache::Load(CMResource& r)
    {
        IResourceLoader* loader;
        IResourceDecompressor* decomp = NULL;
        std::shared_ptr<IResHandle> handle = NULL;

        //DEBUG_OUT_A("Loading '%s' ... \n", r.m_name.c_str());

        std::vector<std::string> elems = util::split(r.m_name, '.');

        if(elems.size() < 2)
        {
            LOG_CRITICAL_ERROR_A("Unknown file format: %s", r.m_name.c_str());
            return handle;
        }

        std::string pattern = elems.back();

        auto it = m_loader.find(pattern);

        if(it == m_loader.end())
        {
            loader = m_defaultLoader;
        } 
        else {
            loader = it->second.get();
        }

        handle = std::shared_ptr<IResHandle>(loader->VCreateHandle());
        handle->VSetResource(r);
        handle->VSetResourceCache(this);

        if(handle) 
        {
            m_handlesList.push_front(handle);
            m_handlesmap[r.m_name] = handle;
        }

        m_lock.Unlock();

        if(m_pFile->VIsCompressed())
        {
            auto compIt = m_decompressor.find(pattern);

            if(compIt == m_decompressor.end())
            {
                decomp = m_defaultDecompressor;
            } 
            else 
            {
                decomp = compIt->second.get();
            }
        }

        LoadFunctionHolder::Load(r, this, m_pFile, loader, decomp, handle);
        m_locks[r.m_name]->Unlock();

        //TODO: throw resource loaded event

        return handle;
    }

    char* ResourceCache::Alloc(uint size)
    {

        if(!MakeRoom(size))
        {
            return NULL;
        }

        char* buffer = new char[size];

        if(!buffer)
        {
            return NULL;
        }

        AllocSilent(size);

        return buffer;
    }

    bool ResourceCache::MakeRoom(uint size)
    {

        if(size > m_maxCacheSize)
        {
            return false;
        }

        while(m_currentCacheSize + size > m_maxCacheSize)
        {
            if(m_handlesList.empty())
            {
                return false;
            }

            FreeOneRessource();
        }

        return true;
    }

    void ResourceCache::FreeOneRessource(void) 
    {
        Free(m_handlesList.back());
    }

    void ResourceCache::Free(std::shared_ptr<IResHandle> handle) 
    {
        m_lock.Lock();

        char* d = handle->VBuffer();
        SAFE_ARRAY_DELETE(d);

        MemoryHasBeenFreed(handle->VSize());

        m_handlesmap.erase(handle->VGetResource().m_name);
        m_handlesList.remove(handle);
        SAFE_DELETE(m_locks[handle->VGetResource().m_name]);
        m_locks.erase(handle->VGetResource().m_name);
        m_lock.Unlock();
    }

    void ResourceCache::Update(std::shared_ptr<IResHandle> handle)
    {
        m_handlesList.remove(handle);
        m_handlesList.push_front(handle);
    }

    void ResourceCache::VFlush(void) 
    {
        while(!m_handlesList.empty())
        {
            Free(m_handlesList.front());
        }
    }

    void ResourceCache::MemoryHasBeenFreed(uint size)
    {
        m_lock.Lock();
        m_currentCacheSize -= size;
        m_lock.Unlock();
    }

    bool ResourceCache::VHasResource(CMResource& r)
    {
        std::vector<std::string> elems = util::split(r.m_name, '.');

        if(elems.size() < 2)
        {
            return false;
        }

        std::string pattern = elems.back();

        auto it = m_loader.find(pattern);

        std::string subFolder("");

        if(it != m_loader.end())
        {
            //return false;
            subFolder = it->second->VSubFolder();
        }

        std::string fileName = m_pFile->VGetName() + "/" + subFolder + r.m_name;
        std::ifstream file(fileName, std::ios::in | std::ios::binary);
        bool good = file.good();
        file.close();
        return good;
    }

    void ResourceCache::AllocSilent(uint size) 
    {
        m_lock.Lock();
        m_currentCacheSize += size;
        m_lock.Unlock();
    }

    ResourceCache::~ResourceCache(void)
    {
        REMOVE_EVENT_LISTENER(this, &ResourceCache::OnResourceChanged, CM_EVENT_RESOURCE_CHANGED);
        while(!m_handlesList.empty())
        {
            FreeOneRessource();
        }
        SAFE_DELETE(m_pFile);
        SAFE_DELETE(m_defaultDecompressor);
        SAFE_DELETE(m_defaultLoader);
    }

    ResHandle::~ResHandle(void) 
    {

    }

    bool ResourceFolder::VOpen(void) 
    {
        DWORD att = GetFileAttributesW(util::string2wstring(this->m_folder).c_str());
        return (att & FILE_ATTRIBUTE_DIRECTORY) != 0;
    }

    int ResourceFolder::VGetRawRessource(const CMResource&  r, char** buffer)
    {
        std::string fileName(m_folder + "/" + r.m_name);
        std::ifstream file(fileName, std::ios::in | std::ios::binary);

        if(!file.is_open())
        {
            LOG_CRITICAL_ERROR_A("File not found: %s", r.m_name.c_str());
            return 0;
        }

        struct stat status;
        stat(fileName.c_str(), &status);
    
        char* tmp = new char[status.st_size];

        file.read(tmp, status.st_size);

        file.close();

        *buffer = tmp;

        return status.st_size;
    }

    bool ResourceFolder::VHasFile(const CMResource& r)
    {
        std::string fileName(m_folder + "/" + r.m_name);
        std::ifstream file(fileName, std::ios::in | std::ios::binary);
        bool open = file.is_open();
        return open;
    }

    int ResourceFolder::VGetNumFiles(void) 
    {
        LOG_CRITICAL_ERROR("not supported");
        return 0;
    }

    std::string ResourceFolder::VGetRessourceName(int num) 
    {
        LOG_CRITICAL_ERROR("not supported");
        return "";
    }

    int DefaultRessourceLoader::VLoadRessource(char* source, uint size, std::shared_ptr<IResHandle>& handle)
    {
        handle->VSetBuffer(source);
        return size;
    }
};

