#pragma once
#include "CMTypes.h"

namespace chimera
{
    class IResHandle
    {
    public:
        virtual CHAR* VBuffer(VOID) = 0;

        virtual UINT VSize(VOID) = 0;

        virtual VOID VSetSize(UINT size) = 0;

        virtual std::string VGetRelativeFilePath(VOID) CONST = 0;

        virtual VOID VSetBuffer(CHAR* buffer) = 0; //dont use this

        virtual IResourceCache* VGetResourceCache(VOID) = 0;

        virtual VOID VSetResourceCache(IResourceCache* cache) = 0;

        virtual VOID VSetResource(CONST CMResource& res) = 0;

        virtual CONST CMResource& VGetResource(VOID) = 0;

        virtual IExtraRessourceData* VGetExtraData(VOID) CONST = 0;

        virtual VOID VSetExtraData(std::unique_ptr<IExtraRessourceData> extraData) = 0;

        virtual BOOL VIsReady(VOID) CONST = 0;

        virtual VOID VSetReady(BOOL ready) = 0;

        virtual ~IResHandle(VOID) {}
    };

    class IMaterial 
    {
    public:
        virtual CONST util::Vec4& VGetSpecular(VOID) CONST = 0;

        virtual CONST util::Vec4& VGetDiffuse(VOID) CONST = 0;

        virtual CONST util::Vec4& VGetAmbient(VOID) CONST = 0;

        virtual FLOAT VGetSpecularExpo(VOID) = 0;

        virtual FLOAT VGetReflectance(VOID) = 0;

        virtual FLOAT VGetTextureScale(VOID) = 0;

        virtual CONST CMResource& VGetTextureDiffuse(VOID) CONST = 0;

        virtual CONST CMResource& VGetTextureNormal(VOID) CONST = 0;

        virtual ~IMaterial(VOID) {}
    };

    class IResourceCache
    {
    public:
        virtual BOOL VInit(UINT mbSize, IResourceFile* resFile) = 0;

        virtual FLOAT VGetWorkload(VOID) CONST = 0;

        virtual size_t VGetNumHandles(VOID) CONST = 0;

        virtual IResourceFile& VGetFile(VOID) = 0;

        virtual VOID VRegisterLoader(std::unique_ptr<IResourceLoader> loader) = 0;

        virtual VOID VRegisterDecompressor(std::unique_ptr<IResourceDecompressor> decomp) = 0;

        virtual std::shared_ptr<IResHandle> VGetHandle(CMResource& r) = 0;

        virtual BOOL VHasResource(CMResource& r) = 0;

        virtual VOID VAppendHandle(std::unique_ptr<IResHandle> handle) = 0;

        virtual std::shared_ptr<IResHandle> VGetHandleAsync(CMResource& r) = 0;

        virtual BOOL VIsLoaded(CMResource& r) = 0;

        virtual VOID VFlush(VOID) = 0;

        virtual ~IResourceCache(VOID) {}
    };

    class IResourceDecompressor 
    {
    public:
        virtual std::string VGetPattern(VOID) = 0;

        virtual BOOL VUseRawFile(VOID) = 0;

        virtual INT VDecompressRessource(CHAR* buffer, INT size, CHAR** dst) = 0;

        virtual ~IResourceDecompressor(VOID) {} 
    };

    class IResourceLoader 
    {
    public:
        virtual CONST std::string& VGetPattern(VOID) = 0;

        virtual INT VLoadRessource(CHAR* source, UINT size, std::shared_ptr<IResHandle> handle) = 0;

        virtual std::unique_ptr<IResHandle> VCreateHandle(VOID) = 0;

        virtual CONST std::string& VSubFolder(VOID) = 0;

        virtual ~IResourceLoader(VOID) {}
    };

    class IResourceFile
    {
    public:
        virtual BOOL VOpen(VOID) = 0;

        virtual INT VGetRawRessource(CONST CMResource&  r, CHAR** buffer) = 0;

        virtual std::string VGetRessourceName(INT num) = 0;

        virtual std::string& VGetName(VOID) = 0;

        virtual INT VGetNumFiles(VOID) = 0;

        virtual BOOL VIsCompressed(VOID) = 0;

        virtual BOOL VHasFile(CONST CMResource& r) = 0;

        virtual ~IResourceFile() {}
    };

    class IExtraRessourceData 
    {
    public:
        virtual ~IExtraRessourceData(VOID) {}
    };

    class IResourceFactory
    {
    public:
        virtual IResourceCache* VCreateCache(VOID) = 0;
    };

    class ResHandle : public IResHandle
    {
    protected:
        CMResource m_resource;
        CHAR* m_data;
        std::unique_ptr<IExtraRessourceData> m_extraData;
        IResourceCache* m_cache;
        BOOL m_isReady;
        UINT m_size;

    public:
        ResHandle(VOID) : m_cache(NULL), m_data(NULL), m_size(0), m_isReady(FALSE) {}

        CHAR* VBuffer(VOID) { return m_data; }

        UINT VSize(VOID) { return m_size; }

        VOID VSetSize(UINT size) { m_size = size; }

        VOID VSetReady(BOOL ready) { m_isReady = ready; }

        std::string VGetRelativeFilePath(VOID) CONST { return m_cache->VGetFile().VGetName() + "/" + m_resource.m_name; }

        //critical only for intern usage //TODO
        VOID VSetBuffer(CHAR* buffer) { m_data = buffer; }

        IResourceCache* VGetResourceCache(VOID) { return m_cache; }

        CMResource& VGetResource(VOID) { return m_resource; }

        VOID VSetResource(CONST CMResource& res) { return m_resource = res; }

        VOID VSetResourceCache(IResourceCache* cache) { m_cache = cache; }

        IExtraRessourceData* VGetExtraData(VOID) CONST { return m_extraData.get(); }

        VOID VSetExtraData(std::unique_ptr<IExtraRessourceData> extraData) { m_extraData = std::move(extraData); }

        BOOL VIsReady(VOID) CONST { return m_isReady; }

        virtual ~ResHandle(VOID);
    };

    struct Triple 
    {
        UINT position;
        UINT texCoord;
        UINT normal;
        ULONG hash;
        UINT index;
        Triple() : position(0), texCoord(0), normal(0), hash(0), index(0) { }

        BOOL Triple::operator==(const Triple& t0)
        {
            return t0.position == position && t0.texCoord == texCoord && t0.normal == normal;
        }

        friend bool operator==(const Triple& t0, const Triple& t1)
        {
            return t0.position == t1.position && t0.texCoord == t1.texCoord && t0.normal == t1.normal;
        }

        friend bool operator<(const Triple& t0, const Triple& t1)
        {
            return t0.position < t1.position;
        }
    };

    struct Face 
    {
        std::vector<Triple> m_triples;
    };

    struct IndexBufferInterval 
    {
        UINT start;
        UINT count;
        UINT material;
        GeometryTopology topo;
        IndexBufferInterval(VOID) : start(0), count(0), material(0), topo(eTopo_Triangles) {}
    };

    class IMesh : public ResHandle
    {
    public:
        virtual CMResource& VGetMaterials(VOID) = 0;

        virtual VOID VAddIndexBufferInterval(UINT start, UINT count, UINT material, GeometryTopology topo) = 0;

        virtual UINT VGetIndexCount(VOID) CONST = 0;

        virtual UINT VGetVertexCount(VOID) CONST = 0;

        virtual UINT VGetVertexStride(VOID) CONST = 0;

        virtual CONST FLOAT* VGetVertices(VOID) CONST = 0;

        virtual CONST std::list<Face>& VGetFaces(VOID) CONST = 0;

        virtual util::AxisAlignedBB& VGetAABB(VOID) = 0;

        virtual CONST UINT* VGetIndices(VOID) CONST = 0;

        virtual VOID VSetIndices(UINT* indices, UINT count) = 0;

        virtual VOID VSetVertices(FLOAT* vertices, UINT count, UINT stride) = 0;

        virtual std::vector<IndexBufferInterval>& VGetIndexBufferIntervals(VOID) = 0;
    };

    //vram

    class IVRamHandle
    {
    public:
        virtual BOOL VCreate(VOID) = 0;

        virtual VOID VDestroy(VOID) = 0;

        virtual UINT VGetByteCount(VOID) CONST = 0;

        virtual BOOL VIsReady(VOID) CONST = 0;

        virtual VOID VUpdate(VOID) = 0;

        virtual VOID VSetResource(CONST VRamResource& res) = 0;

        virtual VOID VSetReady(BOOL ready) = 0;

        virtual LONG VGetLastUsageTime(VOID) CONST = 0;

        virtual CONST VRamResource& VGetResource(VOID) = 0;

        virtual ~IVRamHandle(VOID) {}
    };

    class IVRamHandleCreator
    {
    public:
        virtual IVRamHandle* VGetHandle(VOID) = 0;
        virtual VOID VCreateHandle(IVRamHandle* handle) = 0;
    };

    class VRamHandle : public IVRamHandle
    {
    protected:
        LONG m_lastUsage;
        BOOL m_ready;
        VRamResource m_resource;

    public:
        VRamHandle(VOID) : m_lastUsage(0), m_ready(FALSE) { }
        
        virtual BOOL VCreate(VOID) = 0;

        virtual VOID VDestroy() = 0;

        virtual UINT VGetByteCount(VOID) CONST = 0;

        VOID VSetResource(CONST VRamResource& res) { m_resource = res; }

        BOOL VIsReady(VOID) CONST { return m_ready; }

        VOID VSetReady(BOOL ready) { m_ready = ready; };

        VOID VUpdate(VOID) { m_lastUsage = clock(); }

        LONG VGetLastUsageTime() CONST { return m_lastUsage; }

        VRamResource& VGetResource(VOID) { return m_resource; }

        virtual ~VRamHandle(VOID) {}
    };

    class IVRamManager
    {
    public:
        virtual UINT VGetCurrentSize(VOID) CONST = 0;

        virtual UINT VGetMaxSize(VOID) CONST = 0;

        virtual FLOAT VGetWorkload(VOID) CONST = 0;

        virtual VOID VUpdate(ULONG millis) = 0;

        virtual VOID VFlush(VOID) = 0;

        virtual VOID VFree(std::shared_ptr<IVRamHandle> ressource) = 0;

        virtual std::shared_ptr<IVRamHandle> VGetHandle(CONST VRamResource& ressource) = 0;

        virtual VOID VRegisterHandleCreator(LPCSTR suffix, IVRamHandleCreator* creator) = 0;

        virtual std::shared_ptr<IVRamHandle> VGetHandleAsync(CONST VRamResource& ressource) = 0;

        virtual VOID VAppendAndCreateHandle(std::shared_ptr<IVRamHandle> handle) = 0;

        virtual ~IVRamManager(VOID) {}
    };

    class IVRamManagerFactory
    {
    public:
        virtual IVRamManager* VCreateVRamManager(VOID) = 0;
    };
}