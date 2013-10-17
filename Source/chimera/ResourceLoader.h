#include "Cache.h"

namespace chimera
{
    class ImageExtraData : public IExtraRessourceData 
    {
    public:
        UINT m_width, m_height;
        PixelFormat m_format;
        ImageExtraData(UINT w, UINT h, PixelFormat format) : m_width(w), m_height(h), m_format(format) {}
        ImageExtraData(ImageExtraData& copy) : m_width(copy.m_width), m_height(copy.m_height), m_format(copy.m_format) {}
    };

    class ImageLoader : public DefaultRessourceLoader 
    {
    public:
        ImageLoader(std::string pattern, std::string subFolder) : DefaultRessourceLoader(subFolder, pattern) 
        {
        }
        INT VLoadRessource(CHAR* source, UINT size, std::shared_ptr<IResHandle> handle);
    };

    class ObjLoader : public DefaultRessourceLoader 
    {
    public:
        ObjLoader(std::string subFolder) : DefaultRessourceLoader(subFolder, "obj")
        {
        }
        INT VLoadRessource(CHAR* source, UINT size, std::shared_ptr<IResHandle> handle);
        std::unique_ptr<IResHandle> VCreateHandle(VOID);
    };

    class MaterialLoader : public DefaultRessourceLoader 
    {
    public:
        MaterialLoader(std::string subFolder) : DefaultRessourceLoader(subFolder, "mtl")
        {
        }
        INT VLoadRessource(CHAR* source, UINT size, std::shared_ptr<IResHandle> handle);
        std::unique_ptr<IResHandle> VCreateHandle(VOID);
    };

    class WaveSoundExtraDatra : public IExtraRessourceData
    {
        friend class WaveLoader;
    public:
        WAVEFORMATEX m_format;
        INT m_lengthMillis;
    };

    class WaveLoader : public DefaultRessourceLoader
    {
    private:
        BOOL ParseWaveFile(CHAR* source, std::shared_ptr<IResHandle> handle, UINT& size);
    public:
        WaveLoader(std::string subFolder) : DefaultRessourceLoader(subFolder, "wav")
        {
        }
        INT VLoadRessource(CHAR* source, UINT size, std::shared_ptr<IResHandle> handle);
    };

    //vram

    class GeometryCreator : public IVRamHandleCreator
    {
    public:
        IVRamHandle* VGetHandle(VOID);
        VOID VCreateHandle(IVRamHandle* handle);
    };

    class TextureCreator : public IVRamHandleCreator
    {
    public:
        IVRamHandle* VGetHandle(VOID);
        VOID VCreateHandle(IVRamHandle* handle);
    };
}