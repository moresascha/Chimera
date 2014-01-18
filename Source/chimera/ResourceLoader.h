#include "Cache.h"

namespace chimera
{
    class ImageExtraData : public IExtraRessourceData 
    {
    public:
        uint m_width, m_height;
        PixelFormat m_format;
        ImageExtraData(uint w, uint h, PixelFormat format) : m_width(w), m_height(h), m_format(format) {}
        ImageExtraData(ImageExtraData& copy) : m_width(copy.m_width), m_height(copy.m_height), m_format(copy.m_format) {}
    };

    class ImageLoader : public DefaultRessourceLoader 
    {
    public:
        ImageLoader(std::string pattern, std::string subFolder) : DefaultRessourceLoader(subFolder, pattern) 
        {
        }
        int VLoadRessource(char* source, uint size, std::shared_ptr<IResHandle>& handle);
    };

    class ObjLoader : public DefaultRessourceLoader 
    {
    public:
        ObjLoader(std::string subFolder) : DefaultRessourceLoader(subFolder, "obj")
        {
        }
        int VLoadRessource(char* source, uint size, std::shared_ptr<IResHandle>& handle);
        std::unique_ptr<IResHandle> VCreateHandle(void);
    };

    class MaterialLoader : public DefaultRessourceLoader 
    {
    public:
        MaterialLoader(std::string subFolder) : DefaultRessourceLoader(subFolder, "mtl")
        {
        }
        int VLoadRessource(char* source, uint size, std::shared_ptr<IResHandle>& handle);
        std::unique_ptr<IResHandle> VCreateHandle(void);
    };

    class WaveSoundExtraDatra : public IExtraRessourceData
    {
        friend class WaveLoader;
    public:
        WAVEFORMATEX m_format;
        int m_lengthMillis;
    };

    class WaveLoader : public DefaultRessourceLoader
    {
    private:
        bool ParseWaveFile(char* source, std::shared_ptr<IResHandle> handle, uint& size);
    public:
        WaveLoader(std::string subFolder) : DefaultRessourceLoader(subFolder, "wav")
        {
        }
        int VLoadRessource(char* source, uint size, std::shared_ptr<IResHandle>& handle);
    };

    //vram

    class GeometryCreator : public IVRamHandleCreator
    {
    public:
        IVRamHandle* VGetHandle(void);
        void VCreateHandle(IVRamHandle* handle);
    };

    class TextureCreator : public IVRamHandleCreator
    {
    public:
        IVRamHandle* VGetHandle(void);
        void VCreateHandle(IVRamHandle* handle);
    };
}