#include "stdafx.h"

namespace chimera
{
    namespace d3d
    {
        class D3DGraphicsFactory : public IGraphicsFactory
        {
        public:
            std::unique_ptr<IRenderer> VCreateRenderer(VOID);

            std::unique_ptr<IRenderScreen> VCreateRenderScreen(VOID) { return NULL; }

            std::unique_ptr<IRenderTarget> VCreateRenderTarget(VOID);

            std::unique_ptr<IGeometry> VCreateGeoemtry(VOID);

            std::unique_ptr<IGraphicsStateFactroy> VCreateStateFactory(VOID);

            std::unique_ptr<IShaderFactory> VCreateShaderFactory(VOID);

            std::unique_ptr<IConstShaderBuffer> VCreateConstShaderBuffer(VOID);

            std::unique_ptr<IDeviceTexture> VCreateTexture(CONST CMTextureDescription* desc);
        };
    }
}