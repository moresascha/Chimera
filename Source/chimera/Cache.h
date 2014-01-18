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

        uint m_maxCacheSize;
        uint m_currentCacheSize;

        std::shared_ptr<IResHandle> Find(CMResource& r);
        void Update(std::shared_ptr<IResHandle> handle);
        std::shared_ptr<IResHandle> Load(CMResource& r);
        void Free(std::shared_ptr<IResHandle> handle);
    
        bool MakeRoom(uint size);
        char* Alloc(uint size);
        void AllocSilent(uint size);
        void FreeOneRessource(void);
        void MemoryHasBeenFreed(uint size);

    public:
        ResourceCache(void);

        bool VInit(uint mbSize, IResourceFile* resFile);

        float VGetWorkload(void) const { return 100.0f * m_currentCacheSize / (float) m_maxCacheSize; }

        size_t VGetNumHandles(void) const { return this->m_handlesList.size(); }

        IResourceFile& VGetFile(void) { return *m_pFile; }

        void VRegisterLoader(std::unique_ptr<IResourceLoader> loader);

        void VRegisterDecompressor(std::unique_ptr<IResourceDecompressor> decomp);

        std::shared_ptr<IResHandle> VGetHandle(CMResource& r);

        bool VHasResource(CMResource& r);

        void VAppendHandle(std::unique_ptr<IResHandle> handle);

        void VGetHandleAsync(CMResource& r, OnResourceLoadedCallback cb = NULL);

        bool VIsLoaded(CMResource& r);

        void VFlush(void);

        void OnResourceChanged(std::shared_ptr<IEvent> data);

        ~ResourceCache(void);
    };

    class DefaultRessourceDecompressor : public IResourceDecompressor 
    {
    public:
        std::string VGetPattern(void) { return "*"; }
        bool VUseRawFile(void) { return true; }
        int VDecompressRessource(char* buffer, int size, char** dst) { return size; };
    };

    class DefaultRessourceLoader : public IResourceLoader 
    {
    private:
        std::string m_pattern;
        std::string m_subFolder;

    public:
        DefaultRessourceLoader(void) 
            : m_pattern("*") , 
            m_subFolder("")
        {
        }
        DefaultRessourceLoader(const std::string& subfolder, const std::string& pattern) : m_subFolder(subfolder), m_pattern(pattern) {} 

        virtual const std::string& VGetPattern(void) { return m_pattern; }

        virtual int VLoadRessource(char* source, uint size, std::shared_ptr<IResHandle>& handle);

        virtual std::unique_ptr<IResHandle> VCreateHandle(void) 
        { 
            return std::move(std::unique_ptr<IResHandle>(new ResHandle())); 
        }

        const std::string& VSubFolder(void) 
        {
            return m_subFolder; 
        }

        virtual ~DefaultRessourceLoader(void) { }
    };

    class ResourceFolder : public IResourceFile 
    {
    private:
        std::string m_folder;

    public:
        ResourceFolder(std::string folder) : m_folder(folder) { }

        bool VOpen(void);

        int VGetRawRessource(const CMResource&  r, char** buffer);

        std::string VGetRessourceName(int num);

        int VGetNumFiles(void);

        bool VHasFile(const CMResource& r);

        std::string& VGetName(void) { return m_folder; }

        bool VIsCompressed(void) { return false; }

        ~ResourceFolder() { }
    };
};