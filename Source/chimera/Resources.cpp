#include "Resources.h"
#include "util.h"
#include <fstream>
#include "Mesh.h"
#include "GameApp.h"
#include "Material.h"
#include <sys/stat.h>
#include <sys/types.h>
#include "Process.h"
#include "GameLogic.h"
namespace tbd 
{

    class LoadFunctionHolder 
    {
    public:
        static VOID Load(Resource& r, ResourceCache* cache, IResourceFile* pFile, IResourceLoader* loader, IResourceDecompressor* decomp, std::shared_ptr<ResHandle> handle)
        {
            CHAR* buffer = NULL;
            tbd::Resource tmp(loader->VSubFolder() + r.m_name);
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

            handle->m_size = size;
            handle->m_isReady = TRUE;

            cache->AllocSilent(size);

        }
    };


    class LoadResourceAsyncProcess : public proc::RealtimeProcess
    {
    private:
        std::shared_ptr<ResHandle> m_handle;
        IResourceFile* m_file;
        Resource m_r;
        IResourceDecompressor* m_decomp;
        IResourceLoader* m_loader;
        ResourceCache* m_cache;

    public:

        LoadResourceAsyncProcess(Resource r, ResourceCache* cache, IResourceFile* file, IResourceLoader* loader, IResourceDecompressor* decomp, std::shared_ptr<ResHandle> handle) 
            : m_handle(handle), m_file(file), m_decomp(decomp), m_r(r), m_cache(cache), m_loader(loader)
        {
        }

        VOID VThreadProc(VOID)
        {
            LoadFunctionHolder::Load(m_r, m_cache, m_file, m_loader, m_decomp, m_handle);
            Succeed();
        }
    };

    ResourceCache::ResourceCache(UINT mbSize, IResourceFile* file) : m_defaultDecompressor(NULL), m_defaultLoader(NULL), m_currentCacheSize(NULL), m_maxCacheSize(1024 * 1024 * mbSize), m_pFile(file) 
    {

    }

    BOOL ResourceCache::Init(VOID) 
    {
        if(m_pFile->VOpen())
        {
            m_defaultDecompressor = new DefaultRessourceDecompressor();
            m_defaultLoader = new DefaultRessourceLoader();
            return TRUE;
        }
        return FALSE;
    }

    VOID ResourceCache::RegisterLoader(std::shared_ptr<IResourceLoader> loader) 
    {
        m_loader[loader->VGetPattern()] = loader;
    }

    VOID ResourceCache::RegisterDecompressor(std::shared_ptr<IResourceDecompressor> decompressor) 
    {
        m_decompressor[decompressor->VGetPattern()] = decompressor;
    }

    VOID ResourceCache::AppendHandle(std::shared_ptr<ResHandle> handle)
    {
        m_lock.Lock();
        Resource& r = handle->GetRessource();
        auto it = m_locks.find(r.m_name);
        if(it == m_locks.end())
        {
            m_locks[r.m_name] = new util::Locker();

            m_handlesList.push_front(handle);
            m_handlesmap[r.m_name] = handle;
            handle->m_cache = this;
            handle->m_isReady = TRUE;
            this->AllocSilent(handle->m_size);
        }
        m_lock.Unlock();
    }

    std::shared_ptr<ResHandle> ResourceCache::GetHandle(Resource& r) 
    {
        return _GetHandle(r, FALSE);
    }

    std::shared_ptr<ResHandle> ResourceCache::GetHandleAsync(Resource& r) 
    {
        return _GetHandle(r, TRUE);
    }

    std::shared_ptr<ResHandle> ResourceCache::_GetHandle(Resource& r, BOOL async) 
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

        std::shared_ptr<ResHandle> handle = Find(r);

        if(handle) 
        {
            if(handle->IsReady())
            {
                m_locks[r.m_name]->Unlock();
                Update(handle);
            } 

            m_lock.Unlock();
            //DEBUG_OUT("returning: " + r.m_name);
            return handle;
        }

        handle = Load(r, async);

        return handle;
    }

    std::shared_ptr<ResHandle> ResourceCache::Find(Resource& r) 
    {
        auto handle = m_handlesmap.find(r.m_name);
        if(handle != m_handlesmap.end()) 
        {
            return handle->second;
        }
        return NULL;
    }

    BOOL ResourceCache::IsLoaded(Resource& r)
    {
        std::shared_ptr<ResHandle> handle = Find(r);
        if(handle)
        {
            return handle->IsReady();
        }
        return FALSE;
    }

    std::shared_ptr<ResHandle> ResourceCache::Load(Resource& r, BOOL async)
    {
        IResourceLoader* loader;
        IResourceDecompressor* decomp = NULL;
        std::shared_ptr<ResHandle> handle;

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

        handle = std::shared_ptr<ResHandle>(loader->VCreateHandle());
        handle->m_ressource = r;
        handle->m_cache = this;

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
            std::shared_ptr<proc::Process> proc = std::shared_ptr<proc::Process>(new LoadResourceAsyncProcess(r, this, m_pFile, loader, decomp, handle));
            app::g_pApp->GetLogic()->AttachProcess(proc);
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
        /*
        m_lock.Lock();
        std::shared_ptr<ResHandle> handle = m_handlesList.back();
        m_handlesList.pop_back();
        m_handlesmap.erase(handle->m_ressource.m_name);
        SAFE_DELETE(m_locks[handle->m_ressource.m_name]);
        m_locks.erase(handle->m_ressource.m_name);
        m_lock.Unlock(); */
        Free(m_handlesList.back());
    }

    VOID ResourceCache::Free(std::shared_ptr<ResHandle> handle) 
    {
        m_lock.Lock();
        m_handlesmap.erase(handle->m_ressource.m_name);
        m_handlesList.remove(handle);
        SAFE_DELETE(m_locks[handle->m_ressource.m_name]);
        m_locks.erase(handle->m_ressource.m_name);
        m_lock.Unlock();
    }

    VOID ResourceCache::Update(std::shared_ptr<ResHandle> handle)
    {
        m_handlesList.remove(handle);
        m_handlesList.push_front(handle);
    }

    VOID ResourceCache::Flush(VOID) 
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

    BOOL ResourceCache::HasResource(Resource& r)
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
        if(m_data) 
        {
            delete[] m_data;
        }
        m_cache->MemoryHasBeenFreed(m_size);
    }

    /*
    BOOL RessourceZip::VOpen(VOID) {
        this->m_pZipFile = new ZipFile();
        if(this->m_pZipFile->Init(this->m_fileName))
        {
            return TRUE;
        }
        return FALSE;
    }

    INT RessourceZip::VGetRawRessourceSize(CONST Ressource&  r) {
        int num = this->m_pZipFile->Find(r.m_name.c_str());

        if(num == -1)
        {
            return -1;
        }

        return m_pZipFile->GetFileLen(num);
    }

    INT RessourceZip::VGetRawRessource(CONST Ressource&  r, CHAR* buffer) {
        INT size = 0;
        INT num = m_pZipFile->Find(r.m_name.c_str());
        if(num != -1)
        {
            size = m_pZipFile->GetFileLen(num);
            m_pZipFile->ReadFile(num, buffer);
        }

        return num;
    }

    INT RessourceZip::VGetNumFiles(VOID) {
        return m_pZipFile->GetNumFiles();
    }

    std::string RessourceZip::VGetRessourceName(INT num) {
        if(m_pZipFile && num>=0 && num <m_pZipFile->GetNumFiles())
        {
            return m_pZipFile->GetFilename(num);
        }
        return "";
    } */

    BOOL ResourceFolder::VOpen(VOID) 
    {
        DWORD att = GetFileAttributesW(util::string2wstring(this->m_folder).c_str());
        return att & FILE_ATTRIBUTE_DIRECTORY;
    }

    INT ResourceFolder::VGetRawRessource(CONST Resource&  r, CHAR** buffer)
    {
        std::string fileName(this->m_folder + "/" + r.m_name);
        std::ifstream file(fileName, std::ios::in | std::ios::binary);
        /*
        INT size = -1;
        std::string content;
        if(!file.is_open())
        {
            LOG_ERROR("File not found: " + r.m_name);
        }
        while(file.good())
        {
            std::string line;
            std::getline(file, line);
            content += line + "\n";
        }
        file.close();
        size = content.size();
        *buffer = new CHAR[content.size()];
        CopyMemory(*buffer, content.c_str(), content.length());
        */
        //content.copy(, 0, content.size());

        if(!file.is_open())
        {
            LOG_CRITICAL_ERROR_A("File not found: %s", r.m_name.c_str());
        }

        struct stat status;
        stat(fileName.c_str(), &status);
    
        CHAR* tmp = new CHAR[status.st_size];

        file.read(tmp, status.st_size);

        file.close();

        *buffer = tmp;

        return status.st_size;
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

    BOOL DefaultRessourceLoader::VLoadRessource(CHAR* source, UINT size, std::shared_ptr<ResHandle> handle)
    {
        handle->SetBuffer(source);
        return size;
    }

};

