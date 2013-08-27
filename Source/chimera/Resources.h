#pragma once
#include "stdafx.h"
#include <algorithm>
#include <locale>
#include <gdiplus.h>
#include "Locker.h"

namespace event
{
    class IEvent;
}

namespace tbd 
{

class ResHandle;
class IResourceFile;
class ResourceCache;
class IResourceLoader;
class IResourceDecompressor;
class Resource
{

public:
    std::string m_name;
    Resource(CONST std::string &name) : m_name(name) 
    {
        std::transform(m_name.begin(), m_name.end(), m_name.begin(), ::tolower);
    }

    Resource(LPCSTR name)
    {
        m_name = name;
        std::transform(m_name.begin(), m_name.end(), m_name.begin(), ::tolower);
    }
    
    Resource(CONST Resource& r)
    {
        this->m_name = r.m_name;
    }

    Resource(VOID) : m_name("unknown") {}

    VOID Resource::operator=(std::string& str) 
    {
        std::transform(str.begin(), str.end(), str.begin(), ::tolower);
        this->m_name = str;
    }

    VOID Resource::operator=(CHAR* chars) 
    {
        std::string str(chars);
        std::transform(str.begin(), str.end(), str.begin(), ::tolower);
        this->m_name = str;
    }

    VOID Resource::operator=(CONST Resource& res)
    {
        m_name = res.m_name;
    }

    ~Resource(VOID) { }
};

class ResourceCache {
    friend class ResHandle;
    friend class LoadFunctionHolder;
private:
    IResourceDecompressor* m_defaultDecompressor;
    IResourceLoader* m_defaultLoader;
    util::Locker m_lock;

protected:
    std::list<std::shared_ptr<ResHandle>> m_handlesList;
    std::map<std::string, std::shared_ptr<ResHandle>> m_handlesmap;
    std::map<std::string, std::shared_ptr<IResourceDecompressor>> m_decompressor;
    std::map<std::string, std::shared_ptr<IResourceLoader>> m_loader;
    std::map<std::string, util::Locker*> m_locks;

    IResourceFile* m_pFile;

    UINT m_maxCacheSize;
    UINT m_currentCacheSize;

    std::shared_ptr<ResHandle> Find(Resource& r);
    VOID Update(std::shared_ptr<ResHandle> handle);
    std::shared_ptr<ResHandle> Load(Resource& r, BOOL async);
    VOID Free(std::shared_ptr<ResHandle> handle);
    
    std::shared_ptr<ResHandle> _GetHandle(Resource& r, BOOL async);

    BOOL MakeRoom(UINT size);
    CHAR* Alloc(UINT size);
    VOID AllocSilent(UINT size);
    VOID FreeOneRessource(VOID);
    VOID MemoryHasBeenFreed(UINT size);

public:
    ResourceCache(UINT mbSize, IResourceFile* resFile);

    BOOL Init(VOID);

    FLOAT GetWorkload(VOID) { return 100.0f * m_currentCacheSize / (FLOAT) m_maxCacheSize; }

    size_t GetNumHandles(VOID) { return this->m_handlesList.size(); }

    IResourceFile& GetFile(VOID) { return *m_pFile; }

    VOID RegisterLoader(std::shared_ptr<IResourceLoader> loader);

    VOID RegisterDecompressor(std::shared_ptr<IResourceDecompressor> decomp);

    std::shared_ptr<ResHandle> GetHandle(Resource& r);

    BOOL HasResource(Resource& r);

    VOID AppendHandle(std::shared_ptr<ResHandle> handle);

    std::shared_ptr<ResHandle> GetHandleAsync(Resource& r);

    BOOL IsLoaded(Resource& r);

    VOID Flush(VOID);

    VOID OnResourceChanged(std::shared_ptr<event::IEvent> data);

    ~ResourceCache(VOID);
};

class IResourceFile {
public:
    virtual BOOL VOpen(VOID) = 0;
    virtual INT VGetRawRessource(CONST Resource&  r, CHAR** buffer) = 0;
    virtual std::string VGetRessourceName(INT num) = 0;
    virtual std::string& VGetName(VOID) = 0;
    virtual INT VGetNumFiles(VOID) = 0;
    virtual BOOL VIsCompressed(VOID) = 0;
    virtual ~IResourceFile() { }
};

class IExtreRessourceData 
{
};

class ResHandle {
    friend class ResourceCache;
    friend class LoadFunctionHolder;
    friend class WatchResourceModification;
protected:
    Resource m_resource;
    CHAR* m_data;
    INT m_size;
    std::shared_ptr<IExtreRessourceData> m_extraData;
    ResourceCache* m_cache;
    BOOL m_isReady;
    
public:
    ResHandle(VOID) : m_cache(NULL), m_data(NULL), m_size(0), m_isReady(FALSE) {}

    inline INT Size(VOID) CONST { return m_size; }

    CHAR* Buffer(VOID) CONST { return m_data; }

    std::string GetFullPath(VOID) { return m_cache->GetFile().VGetName() + "/" + this->m_resource.m_name; }

    //critical only for intern usage //TODO
    VOID SetBuffer(CHAR* buffer) { this->m_data = buffer; }

    ResourceCache* GetResourceCache(VOID) { return m_cache; }

    Resource& GetResource(VOID) { return this->m_resource; }

    std::shared_ptr<IExtreRessourceData> GetExtraData(VOID) CONST { return m_extraData; }

    VOID SetExtraData(std::shared_ptr<IExtreRessourceData> extraData) { this->m_extraData = extraData; }

    BOOL IsReady(VOID) CONST { return m_isReady; }

    VOID Update(VOID)
    {
        //std::shared_ptr<ResHandle> t(this); //TODO
       // m_cache->Update(t);
    }

    virtual ~ResHandle(VOID);
};

class IResourceDecompressor {
public:
    virtual std::string VGetPattern(VOID) = 0;
    virtual BOOL VUseRawFile(VOID) = 0;
    virtual INT VDecompressRessource(CHAR* buffer, INT size, CHAR** dst) = 0;
};

class DefaultRessourceDecompressor : public IResourceDecompressor {
public:
    std::string VGetPattern(VOID) { return "*"; }
    BOOL VUseRawFile(VOID) { return TRUE; }
    INT VDecompressRessource(CHAR* buffer, INT size, CHAR** dst) { return size; };
};

class IResourceLoader {
public:
    virtual CONST std::string& VGetPattern(VOID) = 0;
    virtual INT VLoadRessource(CHAR* source, UINT size, std::shared_ptr<ResHandle> handle) = 0;
    virtual ResHandle* VCreateHandle(VOID) = 0;
    virtual CONST std::string& VSubFolder(VOID) = 0;
    virtual ~IResourceLoader(VOID) {}
};

class DefaultRessourceLoader : public IResourceLoader {
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
    virtual INT VLoadRessource(CHAR* source, UINT size, std::shared_ptr<ResHandle> handle);
    virtual ResHandle* VCreateHandle(VOID) { return new ResHandle(); }
    CONST std::string& VSubFolder(VOID) { return m_subFolder; }
    virtual ~DefaultRessourceLoader(VOID) { }
};

class ImageExtraData : public IExtreRessourceData 
{
public:
    UINT m_width, m_height;
    Gdiplus::PixelFormat m_format;
    ImageExtraData(UINT w, UINT h, Gdiplus::PixelFormat format) : m_width(w), m_height(h), m_format(format) {}
    ImageExtraData(ImageExtraData& copy) : m_width(copy.m_width), m_height(copy.m_height), m_format(copy.m_format) {}
};

class ImageLoader : public DefaultRessourceLoader 
{
public:
    ImageLoader(std::string pattern, std::string subFolder) : DefaultRessourceLoader(subFolder, pattern) 
    {
    }
    INT VLoadRessource(CHAR* source, UINT size, std::shared_ptr<ResHandle> handle);
};

class ObjLoader : public DefaultRessourceLoader 
{
public:
    ObjLoader(std::string subFolder) : DefaultRessourceLoader(subFolder, "obj")
    {
    }
    INT VLoadRessource(CHAR* source, UINT size, std::shared_ptr<ResHandle> handle);
    ResHandle* VCreateHandle(VOID);
};

class MaterialLoader : public DefaultRessourceLoader 
{
public:
    MaterialLoader(std::string subFolder) : DefaultRessourceLoader(subFolder, "mtl")
    {
    }
    INT VLoadRessource(CHAR* source, UINT size, std::shared_ptr<ResHandle> handle);
    ResHandle* VCreateHandle(VOID);
};

class WaveSoundExtraDatra : public IExtreRessourceData
{
    friend class WaveLoader;
public:
    WAVEFORMATEX m_format;
    INT m_lengthMillis;
};

class WaveLoader : public DefaultRessourceLoader
{
private:
    BOOL ParseWaveFile(CHAR* source, std::shared_ptr<ResHandle> handle, UINT& size);
public:
    WaveLoader(std::string subFolder) : DefaultRessourceLoader(subFolder, "wav")
    {
    }
    INT VLoadRessource(CHAR* source, UINT size, std::shared_ptr<ResHandle> handle);
};

/*
class RessourceZip : public IRessourceFile {
private:
    ZipFile* m_pZipFile;
    std::wstring m_fileName;
public:
    RessourceZip(std::wstring fileName) : m_fileName(fileName) { }
    virtual BOOL VOpen(VOID);
    virtual INT VGetRawRessource(CONST Ressource&  r, CHAR* buffer);
    virtual INT VGetRawRessourceSize(CONST Ressource&  r);
    virtual std::string VGetRessourceName(INT num);
    virtual INT VGetNumFiles(VOID);

    virtual ~RessourceZip() {
        delete m_pZipFile;
    }
}; */


class ResourceFolder : public IResourceFile {
private:
    std::string m_folder;
public:
    ResourceFolder(std::string folder) : m_folder(folder) { }
    BOOL VOpen(VOID);
    INT VGetRawRessource(CONST Resource&  r, CHAR** buffer);
    std::string VGetRessourceName(INT num);
    INT VGetNumFiles(VOID);
    std::string& VGetName(VOID) { return m_folder; }
    BOOL VIsCompressed(VOID) { return FALSE; }
    ~ResourceFolder() { }
};

};