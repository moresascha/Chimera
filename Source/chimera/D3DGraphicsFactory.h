#include "stdafx.h"

namespace chimera
{
    namespace d3d
    {
        class D3DGraphicsFactory : public IGraphicsFactory
        {
        public:
            std::unique_ptr<IRenderer> VCreateRenderer(void);

            std::unique_ptr<IRenderScreen> VCreateRenderScreen(void) { return NULL; }

            std::unique_ptr<IRenderTarget> VCreateRenderTarget(void);

            std::unique_ptr<IGeometry> VCreateGeoemtry(void);

            std::unique_ptr<IGraphicsStateFactroy> VCreateStateFactory(void);

            std::unique_ptr<IShaderFactory> VCreateShaderFactory(void);

            std::unique_ptr<IConstShaderBuffer> VCreateConstShaderBuffer(void);
            
            std::unique_ptr<IVertexBuffer> VCreateVertexBuffer(void);

            std::unique_ptr<IDeviceBuffer> VCreateIndexBuffer(void);

            std::unique_ptr<IDeviceTexture> VCreateTexture(const CMTextureDescription* desc);
        };
    }
}