#pragma once
#include "D3DGraphicsFactory.h"
#include "D3DConstBuffer.h"
#include "D3DGeometry.h"
#include "D3DRenderer.h"
#include "D3DRenderTarget.h"
#include "D3DShaderProgram.h"
#include "D3DTexture.h"
#include "Process.h"

#ifdef _DEBUG
#define ADD_SHADER_WATCHER \
    std::string __P = CmGetApp()->VGetConfig()->VGetString("sShaderPath"); \
    std::wstring _PP(__P.begin(), __P.end()); \
    CmGetApp()->VGetLogic()->VGetProcessManager()->VAttach(std::shared_ptr<IProcess>(new WatchShaderFileModificationProcess(s, desc->file, _PP.c_str())))
#else
#define ADD_SHADER_WATCHER
#endif

namespace chimera
{
    namespace d3d
    {
        class D3DBlendState : public IBlendState
        {
        private:
            ID3D11BlendState* m_pState;
        public:
            D3DBlendState(CONST BlendStateDesc& dd) : m_pState(NULL)
            {
                D3D11_BLEND_DESC desc;
                memcpy(&desc, &dd, sizeof(D3D11_BLEND_DESC));
                
                D3D_SAVE_CALL(chimera::d3d::GetDevice()->CreateBlendState(&desc, &m_pState));
            }

            VOID* VGetDevicePtr(VOID)
            {
                return m_pState;
            }

            ~D3DBlendState(VOID)
            {
               SAFE_RELEASE(m_pState);
            }
        };

        class D3DRasterState : public IRasterState
        {
        private:
            ID3D11RasterizerState* m_pState;
        public:
            D3DRasterState(CONST RasterStateDesc& dd) : m_pState(NULL)
            {
                D3D11_RASTERIZER_DESC desc;
                desc.AntialiasedLineEnable = dd.AntialiasedLineEnable;
                desc.CullMode = (D3D11_CULL_MODE)dd.CullMode;
                desc.DepthBias = dd.DepthBias;
                desc.DepthBiasClamp = dd.DepthBiasClamp;
                desc.DepthClipEnable = dd.DepthClipEnable;
                desc.FillMode = (D3D11_FILL_MODE)dd.FillMode;
                desc.FrontCounterClockwise = dd.FrontCounterClockwise;
                desc.MultisampleEnable = dd.MultisampleEnable;
                desc.ScissorEnable = dd.ScissorEnable;
                desc.SlopeScaledDepthBias = dd.SlopeScaledDepthBias;

                D3D_SAVE_CALL(chimera::d3d::GetDevice()->CreateRasterizerState(&desc, &m_pState));
            }

            VOID* VGetDevicePtr(VOID)
            {
                return m_pState;
            }

            ~D3DRasterState(VOID)
            {
                SAFE_RELEASE(m_pState);
            }
        };

        class D3DDepthStencilState : public IDepthStencilState
        {
        private:
            ID3D11DepthStencilState* m_pState;
        public:
            D3DDepthStencilState(CONST DepthStencilStateDesc& dd) : m_pState(NULL)
            {
                D3D11_DEPTH_STENCIL_DESC desc;
                memcpy(&desc, &dd, sizeof(D3D11_DEPTH_STENCIL_DESC));

                D3D_SAVE_CALL(chimera::d3d::GetDevice()->CreateDepthStencilState(&desc, &m_pState));
            }

            VOID* VGetDevicePtr(VOID)
            {
                return m_pState;
            }

            ~D3DDepthStencilState(VOID)
            {
                SAFE_RELEASE(m_pState);
            }
        };

        class D3DStateFactory : public IGraphicsStateFactroy
        {
        public:
            IBlendState* VCreateBlendState(CONST BlendStateDesc* desc) 
            {
                return new D3DBlendState(*desc);
            }

            IRasterState* VCreateRasterState(CONST RasterStateDesc* desc) 
            {
                return new D3DRasterState(*desc);
            }

            IDepthStencilState* VCreateDepthStencilState(CONST DepthStencilStateDesc* desc) 
            {
                return new D3DDepthStencilState(*desc);
            }
        };

        template <class T>
        T* CreateShader(LPCTSTR file, LPCSTR function)
        {
            std::string path = CmGetApp()->VGetConfig()->VGetString(SHADER_PATH);
            std::wstring wpath = std::wstring(path.begin(), path.end()) + file;
            T* shader = new T(wpath.c_str(), function);

            ErrorLog log;
            if(!shader->VCompile(&log))
            {
                LOG_CRITICAL_ERROR(log.c_str());
            }

            return shader;
        }

        class D3DShaderFactory : public IShaderFactory
        {
        public:
            IShaderProgram* VCreateShaderProgram(VOID)
            {
                return new chimera::d3d::ShaderProgram();
            }

            IShader* VCreateFragmentShader(CONST CMShaderDescription* desc)
            {
                IShader* s = CreateShader<PixelShader>(desc->file, desc->function);

                ADD_SHADER_WATCHER;

                return s;
            }

            IShader* VCreateVertexShader(CONST CMVertexShaderDescription* desc)
            {
                VertexShader* s = CreateShader<VertexShader>(desc->file, desc->function);
                TBD_FOR_INT(desc->layoutCount)
                {
                    CONST CMVertexInputLayout& layout = desc->inputLayout[i];
                    if(layout.instanced)
                    {
                        s->SetInputAttrInstanced(layout.name, layout.position, layout.slot, GetD3DFormatFromCMFormat(layout.format));
                    }
                    else
                    {
                        s->SetInputAttr(layout.name, layout.position, layout.slot, GetD3DFormatFromCMFormat(layout.format));
                    }
                }
                s->GenerateLayout();
                
                ADD_SHADER_WATCHER;
                
                return s;
            }

            IShader* VCreateGeometryShader(CONST CMShaderDescription* desc)
            {
                IShader* s = CreateShader<GeometryShader>(desc->file, desc->function);

                ADD_SHADER_WATCHER;

                return s;
            }
        };

        std::unique_ptr<IRenderTarget> D3DGraphicsFactory::VCreateRenderTarget(VOID) { return std::unique_ptr<IRenderTarget>(new chimera::d3d::RenderTarget()); }

        std::unique_ptr<IGeometry> D3DGraphicsFactory::VCreateGeoemtry(VOID) { return std::unique_ptr<IGeometry>(new chimera::d3d::Geometry()); }

        std::unique_ptr<IRenderer> D3DGraphicsFactory::VCreateRenderer(VOID) { return std::unique_ptr<IRenderer>(new chimera::d3d::Renderer()); }

        std::unique_ptr<IShaderFactory> D3DGraphicsFactory::VCreateShaderFactory(VOID) { return std::unique_ptr<IShaderFactory>(new D3DShaderFactory()); }

        std::unique_ptr<IConstShaderBuffer> D3DGraphicsFactory::VCreateConstShaderBuffer(VOID) { return std::unique_ptr<IConstShaderBuffer>(new ConstBuffer()); }

        std::unique_ptr<IVertexBuffer> D3DGraphicsFactory::VCreateVertexBuffer(VOID) { return std::unique_ptr<IVertexBuffer>(new VertexBuffer()); }
        
        std::unique_ptr<IDeviceBuffer> D3DGraphicsFactory::VCreateIndexBuffer(VOID) { return std::unique_ptr<IDeviceBuffer>(new IndexBuffer()); }

        std::unique_ptr<IDeviceTexture> D3DGraphicsFactory::VCreateTexture(CONST CMTextureDescription* desc)
        {
            chimera::d3d::Texture2D* texture = new chimera::d3d::Texture2D();

            texture->SetBindflags(D3D11_BIND_SHADER_RESOURCE);

            texture->SetFormat(DXGI_FORMAT_R8G8B8A8_UNORM);

            texture->SetWidth(desc->width);

            texture->SetHeight(desc->height);

            texture->SetMipMapLevels(0);

            texture->SetSamplerCount(1);

            texture->SetSamplerQuality(0);

            texture->SetArraySize(1);

            texture->SetMicsFlags(D3D11_RESOURCE_MISC_GENERATE_MIPS);

            texture->SetData(desc->data);

            return std::unique_ptr<IDeviceTexture>(texture);
        }

        std::unique_ptr<IGraphicsStateFactroy> D3DGraphicsFactory::VCreateStateFactory(VOID)
        {
            return std::unique_ptr<IGraphicsStateFactroy>(new D3DStateFactory());
        }
    }
}