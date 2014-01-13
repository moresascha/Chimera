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

        VOID VOnFileModification(VOID)
        {
            QUEUE_EVENT_TSAVE(new chimera::ResourceChangedEvent(m_resource));
            DEBUG_OUT_A("change on Res: %s\n", m_resource.c_str());
            Succeed();
        }
    };

    class LoadFunctionHolder 
    {
    public:
        static VOID Load(CMResource& r, ResourceCache* cache, IResourceFile* pFile, IResourceLoader* loader, IResourceDecompressor* decomp, std::shared_ptr<IResHandle> handle)
        {
            CHAR* buffer = NULL;
            chimera::CMResource tmp(loader->VSubFolder() + r.m_name);
            UINT size = pFile->VGetRawRessource(tmp, &buffer);

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
                CHAR* dst = NULL;
                UINT newSize = decomp->VDecompressRessource(buffer, size, &dst);
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

            handle->VSetReady(TRUE);
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

    class LoadResourceAsyncProcess : public RealtimeProcess
    {
    private:
        std::shared_ptr<IResHandle> m_handle;
        IResourceFile* m_file;
        CMResource m_r;
        IResourceDecompressor* m_decomp;
        IResourceLoader* m_loader;
        ResourceCache* m_cache;

    public:

        LoadResourceAsyncProcess(CMResource r, ResourceCache* cache, IResourceFile* file, IResourceLoader* loader, IResourceDecompressor* decomp, std::shared_ptr<IResHandle> handle) 
            : m_handle(handle), m_file(file), m_decomp(decomp), m_r(r), m_cache(cache), m_loader(loader)
        {
        }

        VOID VThreadProc(VOID)
        {
            LoadFunctionHolder::Load(m_r, m_cache, m_file, m_loader, m_decomp, m_handle);
            Succeed();
        }
    };

    ResourceCache::ResourceCache(VOID) : m_defaultDecompressor(NULL), m_defaultLoader(NULL), m_currentCacheSize(NULL)
    {
    }

    VOID ResourceCache::OnResourceChanged(IEventPtr data)
    {
        std::shared_ptr<ResourceChangedEvent> event = std::static_pointer_cast<ResourceChangedEvent>(data);
        std::shared_ptr<IResHandle> h = Find(CMResource(event->m_resource));
        if(h)
        {
            h->VSetReady(FALSE);
        }
    }

    BOOL ResourceCache::VInit(UINT mbSize, IResourceFile* resFile) 
    {
         m_maxCacheSize = 1024 * 1024 * mbSize;
         m_pFile = resFile; 
        if(m_pFile->VOpen())
        {
            m_defaultDecompressor = new DefaultRessourceDecompressor();
            m_defaultLoader = new DefaultRessourceLoader();
            return TRUE;
        }
        return FALSE;
    }

    VOID ResourceCache::VRegisterLoader(std::unique_ptr<IResourceLoader> loader) 
    {
        m_loader[loader->VGetPattern()] = std::move(loader);
    }

    VOID ResourceCache::VRegisterDecompressor(std::unique_ptr<IResourceDecompressor> decompressor) 
    {
        m_decompressor[decompressor->VGetPattern()] = std::move(decompressor);
    }

    VOID ResourceCache::VAppendHandle(std::unique_ptr<IResHandle> handle)
    {
        m_lock.Lock();
        CONST CMResource& r = handle->VGetResource();
        auto it = m_locks.find(r.m_name);
        if(it == m_locks.end())
        {
            m_locks[r.m_name] = new util::Locker();

            std::shared_ptr<IResHandle> shrdHanel(std::move(handle));
            m_handlesList.push_front(shrdHanel);

            m_handlesmap[r.m_name] = shrdHanel;

            shrdHanel->VSetResourceCache(this);
            shrdHanel->VSetReady(TRUE);
            AllocSilent(shrdHanel->VSize());
        }
        m_lock.Unlock();
    }

    std::shared_ptr<IResHandle> ResourceCache::VGetHandle(CMResource& r) 
    {
        return _GetHandle(r, FALSE);
    }

    std::shared_ptr<IResHandle> ResourceCache::VGetHandleAsync(CMResource& r) 
    {
        return _GetHandle(r, TRUE);
    }

    std::shared_ptr<IResHandle> ResourceCache::_GetHandle(CMResource& r, BOOL async) 
    {
        m_lock.Lock();
    
        auto it = m_locks.find(r.m_name.c_str());

        if(it == m_locks.end())
        {
            m_locks[r.m_name] = new util::Locker();
        }

        m_lock.Unlock();

        if(!async)
        {
            //DEBUG_OUT_A("waiting for %s %d", r.m_name.c_str(), GetCurrentThreadId());
            m_locks[r.m_name]->Lock();
            //DEBUG_OUT_A("waiting done %s %d", r.m_name.c_str(), GetCurrentThreadId());
        }

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
            CHAR* d = handle->VBuffer();
            SAFE_ARRAY_DELETE(d);
            MemoryHasBeenFreed(handle->VSize());
        }

        handle = Load(r, async);

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

    BOOL ResourceCache::VIsLoaded(CMResource& r)
    {
        std::shared_ptr<IResHandle> handle = Find(r);
        if(handle)
        {
            return handle->VIsReady();
        }
        return FALSE;
    }

    std::shared_ptr<IResHandle> ResourceCache::Load(CMResource& r, BOOL async)
    {
        IResourceLoader* loader;
        IResourceDecompressor* decomp = NULL;
        std::shared_ptr<IResHandle> handle = NULL;

        DEBUG_OUT_A("Loading '%s' ... \n", r.m_name.c_str());

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
        if(async)
        {
            std::unique_ptr<IProcess> proc = std::unique_ptr<IProcess>(new LoadResourceAsyncProcess(r, this, m_pFile, loader, decomp, handle));
            CmGetApp()->VGetLogic()->VGetProcessManager()->VAttach(std::move(proc));
            LOG_CRITICAL_ERROR("this might not work with critical sections");
        }
        else
        {
            LoadFunctionHolder::Load(r, this, m_pFile, loader, decomp, handle);
            m_locks[r.m_name]->Unlock();
        }

        //TODO: throw resource loaded event

        return handle;
    }

    CHAR* ResourceCache::Alloc(UINT size)
    {

        if(!MakeRoom(size))
        {
            return NULL;
        }

        CHAR* buffer = new CHAR[size];

        if(!buffer)
        {
            return NULL;
        }

        AllocSilent(size);

        return buffer;
    }

    BOOL ResourceCache::MakeRoom(UINT size)
    {

        if(size > m_maxCacheSize)
        {
            return FALSE;
        }

        while(m_currentCacheSize + size > m_maxCacheSize)
        {
            if(m_handlesList.empty())
            {
                return FALSE;
            }

            FreeOneRessource();
        }

        return TRUE;
    }

    VOID ResourceCache::FreeOneRessource(VOID) 
    {
        Free(m_handlesList.back());
    }

    VOID ResourceCache::Free(std::shared_ptr<IResHandle> handle) 
    {
        m_lock.Lock();

        CHAR* d = handle->VBuffer();
        SAFE_ARRAY_DELETE(d);

        MemoryHasBeenFreed(handle->VSize());

        m_handlesmap.erase(handle->VGetResource().m_name);
        m_handlesList.remove(handle);
        SAFE_DELETE(m_locks[handle->VGetResource().m_name]);
        m_locks.erase(handle->VGetResource().m_name);
        m_lock.Unlock();
    }

    VOID ResourceCache::Update(std::shared_ptr<IResHandle> handle)
    {
        m_handlesList.remove(handle);
        m_handlesList.push_front(handle);
    }

    VOID ResourceCache::VFlush(VOID) 
    {
        while(!m_handlesList.empty())
        {
            Free(m_handlesList.front());
        }
    }

    VOID ResourceCache::MemoryHasBeenFreed(UINT size)
    {
        m_lock.Lock();
        m_currentCacheSize -= size;
        m_lock.Unlock();
    }

    BOOL ResourceCache::VHasResource(CMResource& r)
    {
        std::vector<std::string> elems = util::split(r.m_name, '.');

        if(elems.size() < 2)
        {
            return FALSE;
        }

        std::string pattern = elems.back();

        auto it = m_loader.find(pattern);

        if(it == m_loader.end())
        {
            return FALSE;
        }
        std::string fileName = m_pFile->VGetName() + "/" + it->second->VSubFolder() + r.m_name;
        std::ifstream file(fileName, std::ios::in | std::ios::binary);
        BOOL good = file.good();
        file.close();
        return good;
    }

    VOID ResourceCache::AllocSilent(UINT size) 
    {
        m_lock.Lock();
        m_currentCacheSize += size;
        m_lock.Unlock();
    }

    ResourceCache::~ResourceCache(VOID)
    {
        while(!m_handlesList.empty())
        {
            FreeOneRessource();
        }
        SAFE_DELETE(m_pFile);
        SAFE_DELETE(m_defaultDecompressor);
        SAFE_DELETE(m_defaultLoader);
    }

    ResHandle::~ResHandle(VOID) 
    {

    }

    BOOL ResourceFolder::VOpen(VOID) 
    {
        DWORD att = GetFileAttributesW(util::string2wstring(this->m_folder).c_str());
        return att & FILE_ATTRIBUTE_DIRECTORY;
    }

    INT ResourceFolder::VGetRawRessource(CONST CMResource&  r, CHAR** buffer)
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
    
        CHAR* tmp = new CHAR[status.st_size];

        file.read(tmp, status.st_size);

        file.close();

        *buffer = tmp;

        return status.st_size;
    }

    BOOL ResourceFolder::VHasFile(CONST CMResource& r)
    {
        std::string fileName(m_folder + "/" + r.m_name);
        std::ifstream file(fileName, std::ios::in | std::ios::binary);
        BOOL open = file.is_open();
        return open;
    }

    INT ResourceFolder::VGetNumFiles(VOID) 
    {
        LOG_CRITICAL_ERROR("not supported");
        return 0;
    }

    std::string ResourceFolder::VGetRessourceName(INT num) 
    {
        LOG_CRITICAL_ERROR("not supported");
        return "";
    }

    INT DefaultRessourceLoader::VLoadRessource(CHAR* source, UINT size, std::shared_ptr<IResHandle> handle)
    {
        handle->VSetBuffer(source);
        return size;
    }
};

