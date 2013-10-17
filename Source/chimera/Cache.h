#pragma once
#include "stdafx.h"
#include <algorithm>
#include <locale>
#include "Locker.h"

namespace chimera 
{
    class ResourceCache : public IResourceCache
    {
        friend class LoadFunctionHolder;
    private:
        IResourceDecompressor* m_defaultDecompressor;
        IResourceLoader* m_defaultLoader;
        util::Locker m_lock;

    protected:
        std::list<std::shared_ptr<IResHandle>> m_handlesList;
        std::map<std::string, std::shared_ptr<IResHandle>> m_handlesmap;
        std::map<std::string, std::unique_ptr<IResourceDecompressor>> m_decompressor;
        std::map<std::string, std::unique_ptr<IResourceLoader>> m_loader;
        std::map<std::string, util::Locker*> m_locks;

        IResourceFile* m_pFile;

        UINT m_maxCacheSize;
        UINT m_currentCacheSize;

        std::shared_ptr<IResHandle> Find(CMResource& r);
        VOID Update(std::shared_ptr<IResHandle> handle);
        std::shared_ptr<IResHandle> Load(CMResource& r, BOOL async);
        VOID Free(std::shared_ptr<IResHandle> handle);
    
        std::shared_ptr<IResHandle> _GetHandle(CMResource& r, BOOL async);

        BOOL MakeRoom(UINT size);
        CHAR* Alloc(UINT size);
        VOID AllocSilent(UINT size);
        VOID FreeOneRessource(VOID);
        VOID MemoryHasBeenFreed(UINT size);

    public:
        ResourceCache(VOID);

        BOOL VInit(UINT mbSize, IResourceFile* resFile);

        FLOAT VGetWorkload(VOID) CONST { return 100.0f * m_currentCacheSize / (FLOAT) m_maxCacheSize; }

        size_t VGetNumHandles(VOID) CONST { return this->m_handlesList.size(); }

        IResourceFile& VGetFile(VOID) { return *m_pFile; }

        VOID VRegisterLoader(std::unique_ptr<IResourceLoader> loader);

        VOID VRegisterDecompressor(std::unique_ptr<IResourceDecompressor> decomp);

        std::shared_ptr<IResHandle> VGetHandle(CMResource& r);

        BOOL VHasResource(CMResource& r);

        VOID VAppendHandle(std::unique_ptr<IResHandle> handle);

        std::shared_ptr<IResHandle> VGetHandleAsync(CMResource& r);

        BOOL VIsLoaded(CMResource& r);

        VOID VFlush(VOID);

        VOID OnResourceChanged(std::shared_ptr<IEvent> data);

        ~ResourceCache(VOID);
    };

    class DefaultRessourceDecompressor : public IResourceDecompressor 
    {
    public:
        std::string VGetPattern(VOID) { return "*"; }
        BOOL VUseRawFile(VOID) { return TRUE; }
        INT VDecompressRessource(CHAR* buffer, INT size, CHAR** dst) { return size; };
    };

    class DefaultRessourceLoader : public IResourceLoader 
    {
    private:
        std::string m_pattern;
        std::string m_subFolder;

    public:
        DefaultRessourceLoader(VOID) 
            : m_pattern("*") , 
            m_subFolder("")
        {
        }
        DefaultRessourceLoader(CONST std::string& subfolder, CONST std::string& pattern) : m_subFolder(subfolder), m_pattern(pattern) {} 

        virtual CONST std::string& VGetPattern(VOID) { return m_pattern; }

        virtual INT VLoadRessource(CHAR* source, UINT size, std::shared_ptr<IResHandle> handle);

        virtual std::unique_ptr<IResHandle> VCreateHandle(VOID) 
        { 
            return std::move(std::unique_ptr<IResHandle>(new ResHandle())); 
        }

        CONST std::string& VSubFolder(VOID) 
        {
            return m_subFolder; 
        }

        virtual ~DefaultRessourceLoader(VOID) { }
    };

    class ResourceFolder : public IResourceFile 
    {
    private:
        std::string m_folder;

    public:
        ResourceFolder(std::string folder) : m_folder(folder) { }

        BOOL VOpen(VOID);

        INT VGetRawRessource(CONST CMResource&  r, CHAR** buffer);

        std::string VGetRessourceName(INT num);

        INT VGetNumFiles(VOID);

        std::string& VGetName(VOID) { return m_folder; }

        BOOL VIsCompressed(VOID) { return FALSE; }

        ~ResourceFolder() { }
    };
};