#pragma once
#include "CMTypes.h"

namespace chimera
{
    class IResHandle
    {
    public:
        virtual char* VBuffer(void) = 0;

        virtual uint VSize(void) = 0;

        virtual void VSetSize(uint size) = 0;

        virtual std::string VGetRelativeFilePath(void) const = 0;

        virtual void VSetBuffer(char* buffer) = 0; //dont use this

        virtual IResourceCache* VGetResourceCache(void) = 0;

        virtual void VSetResourceCache(IResourceCache* cache) = 0;

        virtual void VSetResource(const CMResource& res) = 0;

        virtual const CMResource& VGetResource(void) = 0;

        virtual IExtraRessourceData* VGetExtraData(void) const = 0;

        virtual void VSetExtraData(std::unique_ptr<IExtraRessourceData> extraData) = 0;

        virtual bool VIsReady(void) const = 0;

        virtual void VSetReady(bool ready) = 0;

        virtual ~IResHandle(void) {}
    };

    typedef void (*OnResourceLoadedCallback)(std::shared_ptr<IResHandle>& handle);

    class IMaterial 
    {
    public:
        virtual const util::Vec4& VGetSpecular(void) const = 0;

        virtual const util::Vec4& VGetDiffuse(void) const = 0;

        virtual const util::Vec4& VGetAmbient(void) const = 0;

        virtual float VGetSpecularExpo(void) = 0;

        virtual float VGetReflectance(void) = 0;

        virtual float VGetTextureScale(void) = 0;

        virtual const CMResource& VGetTextureDiffuse(void) const = 0;

        virtual const CMResource& VGetTextureNormal(void) const = 0;

        virtual ~IMaterial(void) {}
    };

    class IResourceCache
    {
    public:
        virtual bool VInit(uint mbSize, IResourceFile* resFile) = 0;

        virtual float VGetWorkload(void) const = 0;

        virtual size_t VGetNumHandles(void) const = 0;

        virtual IResourceFile& VGetFile(void) = 0;

        virtual void VRegisterLoader(std::unique_ptr<IResourceLoader> loader) = 0;

        virtual void VRegisterDecompressor(std::unique_ptr<IResourceDecompressor> decomp) = 0;

        virtual std::shared_ptr<IResHandle> VGetHandle(CMResource& r) = 0;

        virtual bool VHasResource(CMResource& r) = 0;

        virtual void VAppendHandle(std::unique_ptr<IResHandle> handle) = 0;

        virtual void VGetHandleAsync(CMResource& r, OnResourceLoadedCallback cb = NULL) = 0;

        virtual bool VIsLoaded(CMResource& r) = 0;

        virtual void VFlush(void) = 0;

        virtual ~IResourceCache(void) {}
    };

    class IResourceDecompressor 
    {
    public:
        virtual std::string VGetPattern(void) = 0;

        virtual bool VUseRawFile(void) = 0;

        virtual int VDecompressRessource(char* buffer, int size, char** dst) = 0;

        virtual ~IResourceDecompressor(void) {} 
    };

    class IResourceLoader 
    {
    public:
        virtual const std::string& VGetPattern(void) = 0;

        virtual int VLoadRessource(char* source, uint size, std::shared_ptr<IResHandle>& handle) = 0;

        virtual std::unique_ptr<IResHandle> VCreateHandle(void) = 0;

        virtual const std::string& VSubFolder(void) = 0;

        virtual ~IResourceLoader(void) {}
    };

    class IResourceFile
    {
    public:
        virtual bool VOpen(void) = 0;

        virtual int VGetRawRessource(const CMResource&  r, char** buffer) = 0;

        virtual std::string VGetRessourceName(int num) = 0;

        virtual std::string& VGetName(void) = 0;

        virtual int VGetNumFiles(void) = 0;

        virtual bool VIsCompressed(void) = 0;

        virtual bool VHasFile(const CMResource& r) = 0;

        virtual ~IResourceFile() {}
    };

    class IExtraRessourceData 
    {
    public:
        virtual ~IExtraRessourceData(void) {}
    };

    class IResourceFactory
    {
    public:
        virtual IResourceCache* VCreateCache(void) = 0;
    };

    class ResHandle : public IResHandle
    {
    protected:
        CMResource m_resource;
        char* m_data;
        std::unique_ptr<IExtraRessourceData> m_extraData;
        IResourceCache* m_cache;
        bool m_isReady;
        uint m_size;

    public:
        ResHandle(void) : m_cache(NULL), m_data(NULL), m_size(0), m_isReady(false) {}

        char* VBuffer(void) { return m_data; }

        uint VSize(void) { return m_size; }

        void VSetSize(uint size) { m_size = size; }

        void VSetReady(bool ready) { m_isReady = ready; }

        std::string VGetRelativeFilePath(void) const { return m_cache->VGetFile().VGetName() + "/" + m_resource.m_name; }

        //critical only for intern usage //TODO
        void VSetBuffer(char* buffer) { m_data = buffer; }

        IResourceCache* VGetResourceCache(void) { return m_cache; }

        CMResource& VGetResource(void) { return m_resource; }

        void VSetResource(const CMResource& res) { return m_resource = res; }

        void VSetResourceCache(IResourceCache* cache) { m_cache = cache; }

        IExtraRessourceData* VGetExtraData(void) const { return m_extraData.get(); }

        void VSetExtraData(std::unique_ptr<IExtraRessourceData> extraData) { m_extraData = std::move(extraData); }

        bool VIsReady(void) const { return m_isReady; }

        virtual ~ResHandle(void);
    };

    struct Triple 
    {
        uint position;
        uint texCoord;
        uint normal;
        ulong hash;
        uint index;
        Triple() : position(0), texCoord(0), normal(0), hash(0), index(0) { }

        bool Triple::operator==(const Triple& t0)
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
        uint start;
        uint count;
        uint material;
        GeometryTopology topo;
        IndexBufferInterval(void) : start(0), count(0), material(0), topo(eTopo_Triangles) {}
    };

    class IMesh
    {
    public:
        virtual CMResource& VGetMaterials(void) = 0;

        virtual void VAddIndexBufferInterval(uint start, uint count, uint material, GeometryTopology topo) = 0;

        virtual uint VGetIndexCount(void) const = 0;

        virtual uint VGetVertexCount(void) const = 0;

        virtual uint VGetVertexStride(void) const = 0;

        virtual const float* VGetVertices(void) const = 0;

        virtual const std::list<Face>& VGetFaces(void) const = 0;

        virtual util::AxisAlignedBB& VGetAABB(void) = 0;

        virtual const uint* VGetIndices(void) const = 0;

        virtual void VSetIndices(uint* indices, uint count) = 0;

        virtual void VSetVertices(float* vertices, uint count, uint stride) = 0;

        virtual std::vector<IndexBufferInterval>& VGetIndexBufferIntervals(void) = 0;
    };

    class IMeshSet : public ResHandle
    {
    public:
        typedef std::map<std::string, std::shared_ptr<IMesh>>::iterator MeshIterator;

        virtual uint VGetMeshCount(void) = 0;

        virtual IMesh* VGetMesh(const std::string& name) = 0;

        virtual IMesh* VGetMesh(uint i) = 0;

        virtual MeshIterator VBegin(void) = 0;

        virtual MeshIterator VEnd(void) = 0;
    };

    //vram

    class IVRamHandle
    {
    public:
        virtual bool VCreate(void) = 0;

        virtual void VDestroy(void) = 0;

        virtual uint VGetByteCount(void) const = 0;

        virtual bool VIsReady(void) const = 0;

        virtual void VUpdate(void) = 0;

        virtual void VSetResource(const VRamResource& res) = 0;

        virtual void VSetReady(bool ready) = 0;

        virtual LONG VGetLastUsageTime(void) const = 0;

        virtual const VRamResource& VGetResource(void) = 0;

        virtual ~IVRamHandle(void) {}
    };

    class IVRamHandleCreator
    {
    public:
        virtual IVRamHandle* VGetHandle(void) = 0;

        virtual void VCreateHandle(IVRamHandle* handle) = 0;
    };

    class VRamHandle : public IVRamHandle
    {
    protected:
        LONG m_lastUsage;
        bool m_ready;
        VRamResource m_resource;

    public:
        VRamHandle(void) : m_lastUsage(0), m_ready(false) { }
        
        virtual bool VCreate(void) = 0;

        virtual void VDestroy() = 0;

        virtual uint VGetByteCount(void) const = 0;

        void VSetResource(const VRamResource& res) { m_resource = res; }

        bool VIsReady(void) const { return m_ready; }

        void VSetReady(bool ready) { m_ready = ready; };

        void VUpdate(void) { m_lastUsage = clock(); }

        LONG VGetLastUsageTime() const { return m_lastUsage; }

        VRamResource& VGetResource(void) { return m_resource; }

        virtual ~VRamHandle(void) {}
    };

    class IVRamManager
    {
    public:
        virtual uint VGetCurrentSize(void) const = 0;

        virtual uint VGetMaxSize(void) const = 0;

        virtual float VGetWorkload(void) const = 0;

        virtual void VUpdate(ulong millis) = 0;

        virtual void VFlush(void) = 0;

        virtual void VFree(std::shared_ptr<IVRamHandle> ressource) = 0;

        virtual std::shared_ptr<IVRamHandle> VGetHandle(const VRamResource& ressource) = 0;

        virtual void VRegisterHandleCreator(LPCSTR suffix, IVRamHandleCreator* creator) = 0;

        virtual std::shared_ptr<IVRamHandle> VGetHandleAsync(const VRamResource& ressource) = 0;

        virtual void VAppendAndCreateHandle(std::shared_ptr<IVRamHandle> handle) = 0;

        virtual ~IVRamManager(void) {}
    };

    class IVRamManagerFactory
    {
    public:
        virtual IVRamManager* VCreateVRamManager(void) = 0;
    };
}