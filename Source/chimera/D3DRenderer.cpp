#include "D3DRenderer.h"
#include "D3DGraphics.h"
#include "D3DRenderTarget.h"
#include "GeometryFactory.h"
#include "D3DGeometry.h"
#include "D3DConstBuffer.h"
#include "Vec4.h"
#include "Vec3.h"
#include "Mat4.h"
#include "ShaderCache.h"
#include "Application.h"
#include "Material.h"

namespace chimera 
{
    namespace d3d
    {

#define VS_Stage 1
#define PS_Stage 2
#define GS_Stage 4

        Renderer::Renderer(VOID) 
            : m_pDefaultRenderTarget(NULL), m_pDefShader(NULL), m_pDefaultRasterState(NULL), m_pDefaultDepthStencilState(NULL), m_pDefaultBlendState(NULL), m_pAlphaBlendState(NULL)
        {
            for(UCHAR i = 0; i < BufferCnt; ++i)
            {
                m_constBuffer[i] = NULL;
            }
            for(UCHAR i = 0; i < MAX_SAMPLER; ++i)
            {
                m_currentSetSampler[i] = NULL;
            }
            Material* mat = new Material();
            mat->m_ambient.Set(0.5f,0.5f,0.5f,1);
            mat->m_diffuse.Set(1, 1, 1, 1);
            mat->m_specular.Set(0, 0, 0, 0);
            mat->m_texScale = 1;
            mat->m_textureDiffuse = chimera::CMResource("default64x64.png");
            m_pDefaultMaterial = mat;

            /*VPushRasterizerState(chimera::g_pRasterizerStateFrontFaceSolid);
            VPushBlendState(chimera::g_pBlendStateNoBlending);*/
        }

        UINT Renderer::VGetHeight(VOID) { return chimera::d3d::g_height; }

        UINT Renderer::VGetWidth(VOID) { return chimera::d3d::g_width; }

        VOID Renderer::VPresent(VOID)
        {
            chimera::d3d::g_pSwapChain->Present(0, 0);
        }

        VOID Renderer::CreateDefaultShader(VOID)
        {
            CMShaderProgramDescription desc;
            desc.fs.file = SCREENQUAD_SHADER_FILE;
            desc.fs.function = SCREENQUAD_SHADER_FS_FUNCTION;

            desc.vs.layoutCount = 2;

            desc.vs.inputLayout[0].format = eFormat_R32G32B32_FLOAT;
            desc.vs.inputLayout[0].instanced = FALSE;
            desc.vs.inputLayout[0].name = "POSITION";
            desc.vs.inputLayout[0].position = 0;
            desc.vs.inputLayout[0].slot = 0;

            desc.vs.inputLayout[1].format = eFormat_R32G32_FLOAT;
            desc.vs.inputLayout[1].instanced = FALSE;
            desc.vs.inputLayout[1].name = "TEXCOORD";
            desc.vs.inputLayout[1].position = 1;
            desc.vs.inputLayout[1].slot = 0;

            desc.vs.file = SCREENQUAD_SHADER_FILE;
            desc.vs.function = SCREENQUAD_SHADER_VS_FUNCTION;

            m_screenQuadProgram = VGetShaderCache()->VCreateShaderProgram(SCREENQUAD_SHADER_NAME, &desc);
        }

        BOOL Renderer::VCreate(CM_WINDOW_CALLBACK cb, CM_INSTANCE instance, LPCWSTR wndTitle, UINT width, UINT height)
        {
            chimera::d3d::Init(cb, instance, wndTitle, width, height);
            m_pShaderCache = std::unique_ptr<IShaderCache>(
                new ShaderCache(std::move(std::unique_ptr<IShaderFactory>(CmGetApp()->VGetHumanView()->VGetGraphicsFactory()->VCreateShaderFactory())))
                );
            chimera::d3d::Geometry::Create();

            BlendStateDesc blendDesc;
            ZeroMemory(&blendDesc, sizeof(BlendStateDesc));
            TBD_FOR_INT(8)
            {
                blendDesc.RenderTarget[i].BlendEnable = FALSE;
                blendDesc.RenderTarget[i].RenderTargetWriteMask = eColorWriteAll;
            }

            std::unique_ptr<IGraphicsStateFactroy> stateFactory = CmGetApp()->VGetHumanView()->VGetGraphicsFactory()->VCreateStateFactory();

            m_pDefaultBlendState = stateFactory->VCreateBlendState(&blendDesc);
            VPushBlendState(m_pDefaultBlendState);

            DepthStencilStateDesc depthStencilDesc;
            ZeroMemory(&depthStencilDesc, sizeof(DepthStencilStateDesc));

            depthStencilDesc.StencilEnable = TRUE;
            depthStencilDesc.StencilWriteMask = 0xFF;
            depthStencilDesc.StencilReadMask = 0xFF;

            depthStencilDesc.FrontFace.StencilFailOp = eStencilOP_Keep;
            depthStencilDesc.FrontFace.StencilDepthFailOp = eStencilOP_Keep;
            depthStencilDesc.FrontFace.StencilPassOp = eStencilOP_Incr;
            depthStencilDesc.FrontFace.StencilFunc = eCompareFunc_Always;

            depthStencilDesc.BackFace.StencilFailOp = eStencilOP_Keep;
            depthStencilDesc.BackFace.StencilDepthFailOp = eStencilOP_Keep;
            depthStencilDesc.BackFace.StencilPassOp = eStencilOP_Incr; 
            depthStencilDesc.BackFace.StencilFunc = eCompareFunc_Always;

            m_pDefaultDepthStencilState = stateFactory->VCreateDepthStencilState(&depthStencilDesc);
            VPushDepthStencilState(m_pDefaultDepthStencilState);

            RasterStateDesc rasterDesc;
            ZeroMemory(&rasterDesc, sizeof(RasterStateDesc));
            rasterDesc.CullMode = eCullMode_Back;;
            rasterDesc.FillMode = eFillMode_Solid;
            rasterDesc.DepthClipEnable = TRUE;
            rasterDesc.FrontCounterClockwise = TRUE;
            rasterDesc.MultisampleEnable = FALSE;
            rasterDesc.AntialiasedLineEnable = FALSE;

            m_pDefaultRasterState = stateFactory->VCreateRasterState(&rasterDesc);
            VPushRasterState(m_pDefaultRasterState);

            ZeroMemory(&blendDesc, sizeof(BlendStateDesc));
            for(int i = 0; i < 8; ++i)
            {
                blendDesc.RenderTarget[i].BlendEnable = TRUE;
                blendDesc.RenderTarget[i].RenderTargetWriteMask = eColorWriteAll;

                blendDesc.RenderTarget[i].BlendOp = eBlendOP_Add;
                blendDesc.RenderTarget[i].BlendOpAlpha = eBlendOP_Add;

                blendDesc.RenderTarget[i].DestBlend = eBlend_InvSrcAlpha;
                blendDesc.RenderTarget[i].DestBlendAlpha = eBlend_InvSrcAlpha;

                blendDesc.RenderTarget[i].SrcBlend = eBlend_SrcAlpha;
                blendDesc.RenderTarget[i].SrcBlendAlpha = eBlend_SrcAlpha;  
            }

            m_pAlphaBlendState = CmGetApp()->VGetHumanView()->VGetGraphicsFactory()->VCreateStateFactory()->VCreateBlendState(&blendDesc);       

            CreateDefaultShader();

            return VOnRestore();
        }

        VOID Renderer::VDestroy(VOID)
        {
            SAFE_DELETE(m_pDefaultRasterState);
            SAFE_DELETE(m_pDefaultBlendState);
            SAFE_DELETE(m_pDefaultDepthStencilState);
            SAFE_DELETE(m_pDefaultMaterial);
            SAFE_DELETE(m_pAlphaBlendState);
            geometryfactroy::Destroy();
            chimera::d3d::Geometry::Destroy();
            Delete();
            SAFE_DELETE(m_pDefShader);
            chimera::d3d::Release();
        }

        BOOL Renderer::VOnRestore(VOID) 
        {
            Delete();
            //m_pDefaultTexture = std::static_pointer_cast<D3DTexture2D>(chimera::g_pApp->VGetHumanView()->VGetVRamManager()->VGetHandle(m_defaultMaterial.m_textureDiffuse));

            if(m_pDefShader)
            {
                m_pDefShader->VOnRestore(chimera::d3d::g_width, chimera::d3d::g_height);
            }
            else
            {
                m_pDefShader = new chimera::d3d::AlbedoBuffer(chimera::d3d::g_width, chimera::d3d::g_height);
            }

            m_backColor[0] = 0; m_backColor[1] = 0; m_backColor[2] = 0; m_backColor[3] = 0;

            m_constBuffer[eModelBuffer] = new ConstBuffer();
            m_constBuffer[eModelBuffer]->VInit(sizeof(_ModelMatrixBuffer));

            m_constBuffer[eViewBuffer] = new ConstBuffer();
            m_constBuffer[eViewBuffer]->VInit(sizeof(_ViewMatrixBuffer));
        
            m_constBuffer[eProjectionBuffer] = new ConstBuffer();
            m_constBuffer[eProjectionBuffer]->VInit(sizeof(_ProjectionMatrixBuffer));

            m_constBuffer[eMaterialBuffer] = new ConstBuffer();
            m_constBuffer[eMaterialBuffer]->VInit(sizeof(_MaterialBuffer));

            m_constBuffer[eLightingBuffer] = new ConstBuffer();
            m_constBuffer[eLightingBuffer]->VInit(sizeof(_LightSettingsBuffer));

            m_constBuffer[eCubeMapViewsBuffer] = new ConstBuffer();
            m_constBuffer[eCubeMapViewsBuffer]->VInit(sizeof(_CubeMapViewsBuffer));

            m_constBuffer[eFontBuffer] = new ConstBuffer();
            m_constBuffer[eFontBuffer]->VInit(8 * sizeof(FLOAT));

            m_constBuffer[eBoundingGeoBuffer] = new ConstBuffer();
            m_constBuffer[eBoundingGeoBuffer]->VInit(4 * sizeof(FLOAT));

            m_constBuffer[eActorIdBuffer] = new ConstBuffer();
            m_constBuffer[eActorIdBuffer]->VInit(4 * sizeof(UINT));

            m_constBuffer[eSelectedActorIdBuffer] = new ConstBuffer();
            m_constBuffer[eSelectedActorIdBuffer]->VInit(4 * sizeof(UINT));

            m_constBuffer[eGuiColorBuffer] = new ConstBuffer();
            m_constBuffer[eGuiColorBuffer]->VInit(4 * sizeof(FLOAT));

            m_constBuffer[eHasNormalMapBuffer] = new ConstBuffer();
            m_constBuffer[eHasNormalMapBuffer]->VInit(4 * sizeof(UINT));

            m_constBuffer[eEnvLightingBuffer] = new ConstBuffer();
            m_constBuffer[eEnvLightingBuffer]->VInit(sizeof(_LightingBuffer));

            ID3D11Buffer* buffer[BufferCnt];
            for(UCHAR i = 0; i < BufferCnt; ++i)
            {
                buffer[i] = (ID3D11Buffer*)m_constBuffer[i]->VGetDevicePtr();
            }

            chimera::d3d::GetContext()->VSSetConstantBuffers(0, BufferCnt, buffer);
            chimera::d3d::GetContext()->PSSetConstantBuffers(0, BufferCnt, buffer);
            chimera::d3d::GetContext()->GSSetConstantBuffers(0, BufferCnt, buffer);

            m_pDefaultTexture = std::static_pointer_cast<IDeviceTexture>(CmGetApp()->VGetHumanView()->VGetVRamManager()->VGetHandle(m_pDefaultMaterial->VGetTextureDiffuse()));
     
            return TRUE;
        }

        VOID* Renderer::VGetDevice(VOID)
        {
            return (VOID*)chimera::d3d::GetDevice();
        }

        VOID Renderer::VResize(UINT w, UINT h)
        {
            chimera::d3d::Resize(w, h);
        }

        VOID Renderer::VSetDefaultTexture(VOID)
        {
            if(m_pDefaultTexture->VIsReady())
            {
                m_pDefaultTexture->VUpdate();
            }
            else
            {
                m_pDefaultTexture = std::static_pointer_cast<IDeviceTexture>(CmGetApp()->VGetHumanView()->VGetVRamManager()->VGetHandle(m_pDefaultMaterial->VGetTextureDiffuse()));
            }
            VSetTexture(eDiffuseColorSampler, m_pDefaultTexture.get());
            VSetNormalMapping(FALSE);
        }

        /*VOID D3DRenderer::SetDefaultRasterizerState(ID3D11RasterizerState* state)
        {
            m_rasterStateStack.Clear();
            PushRasterizerState(state);
        } */

        VOID Renderer::VSetDefaultMaterial(VOID)
        {
            VPushMaterial(*m_pDefaultMaterial);
        }

        /*
        VOID D3DRenderer::TransferBuffer(std::shared_ptr<d3d::ShaderProgram> program, UINT destinationSlot, UINT sourceSlot) CONST 
        {
            ID3D11Buffer* buffer[4];
            buffer[VIEW_MATRIX_SLOT] = m_viewBuffer->GetBuffer();
            buffer[PROJECTION_MATRIX_SLOT] = m_projBuffer->GetBuffer();
            buffer[MODEL_MATRIX_SLOT] = m_modelBuffer->GetBuffer();
            buffer[MATERIAL_SLOT] = m_materialBuffer->GetBuffer();

            if(destinationSlot == -1)
            {
                program->SetConstBuffers(0, buffer, 4);
            }
            else
            {
                program->SetConstBuffers(destinationSlot, &buffer[sourceSlot], 1);
            }
        } */

        VOID Renderer::VPreRender(VOID) 
        {
            m_pDefShader->VUnbindRenderTargets();
            VSetTexture(eAmbientMaterialSpecGSampler, m_pDefShader->VGetRenderTarget(eDiff_AmbientMaterialSpecGTarget)->VGetTexture());
            VSetTexture(eDiffuseColorSpecBSampler, m_pDefShader->VGetRenderTarget(eDiff_DiffuseColorSpecBTarget)->VGetTexture());
            VSetTexture(eDiffuseMaterialSpecRSampler, m_pDefShader->VGetRenderTarget(eDiff_DiffuseMaterialSpecRTarget)->VGetTexture());
            VSetTexture(eNormalsSampler, m_pDefShader->VGetRenderTarget(eDiff_NormalsTarget)->VGetTexture());
            VSetTexture(eWorldPositionSampler, m_pDefShader->VGetRenderTarget(eDiff_WorldPositionTarget)->VGetTexture());
            VSetTexture(eNormalColorSampler, m_pDefShader->VGetRenderTarget(eDiff_ReflectionStrTarget)->VGetTexture());
        }

        VOID Renderer::VPostRender(VOID) 
        {
            VSetTexture(eAmbientMaterialSpecGSampler, NULL);
            VSetTexture(eDiffuseColorSpecBSampler, NULL);
            VSetTexture(eDiffuseMaterialSpecRSampler, NULL);
            VSetTexture(eNormalsSampler, NULL);
            VSetTexture(eWorldPositionSampler, NULL);
            VSetTexture(eNormalColorSampler, NULL);
            m_pDefShader->VClearAndBindRenderTargets();
        }

        VOID Renderer::VSetNormalMapping(BOOL map)
        {
            IConstShaderBuffer* buffer = VGetConstShaderBuffer(chimera::eHasNormalMapBuffer);
            FLOAT* data = (FLOAT*)buffer->VMap();
            data[0] = map ? 1.0f : 0.0f;
            data[1] = map ? 1.0f : 0.0f;
            data[2] = map ? 1.0f : 0.0f;
            data[3] = map ? 1.0f : 0.0f;
            buffer->VUnmap();
        }

        VOID Renderer::VPushCurrentRenderTarget(IRenderTarget* rt)
        {
            m_pDefaultRenderTarget = rt;
        }

        VOID Renderer::VPopCurrentRenderTarget(VOID)
        {

        }

        VOID Renderer::VClearAndBindBackBuffer(VOID)
        {
            VBindBackBuffer();
            chimera::d3d::ClearBackBuffer(m_backColor);
        }

        VOID Renderer::VBindBackBuffer(VOID)
        {
            chimera::d3d::SetDefaultViewPort();
            chimera::d3d::BindBackbuffer();
        }

        IRenderTarget* Renderer::VGetCurrentRenderTarget(VOID)
        {
            return m_pDefaultRenderTarget;
        }

        VOID Renderer::VSetWorldTransform(CONST util::Mat4& mat) 
        {
            m_constBuffer[eModelBuffer]->VSetFromMatrix(mat);
        }

        VOID Renderer::VPushWorldTransform(CONST util::Mat4& mat) 
        {
            m_constBuffer[eModelBuffer]->VSetFromMatrix(mat);
        }

        VOID Renderer::VPopWorldTransform(VOID)
        {

        }

        VOID Renderer::VSetViewTransform(CONST util::Mat4& mat, CONST util::Mat4& invMat, CONST util::Vec3& eyePos)
        {
            m_viewMatrixStack.Clear();
            _ViewMatrixBuffer buffer;
            buffer.m_view = mat.m_m;
            buffer.m_invView = invMat.m_m;
            buffer.m_eyePos.x = eyePos.m_v.x;
            buffer.m_eyePos.y = eyePos.m_v.y;
            buffer.m_eyePos.z = eyePos.m_v.z;
            buffer.m_eyePos.w = 0;

            _ViewMatrixBuffer* vb = (_ViewMatrixBuffer*)m_constBuffer[eViewBuffer]->VMap();
            *vb = buffer;
            m_constBuffer[eViewBuffer]->VUnmap();
            //m_viewBuffer->SetFromMatrix(mat);
            m_viewMatrixStack.Push(buffer);
            chimera::CmGetApp()->VGetLogic()->VGetHumanView()->VGetSceneGraph()->VResetVisibility();
        }

        VOID Renderer::VPushViewTransform(CONST util::Mat4& mat, CONST util::Mat4& invMat, CONST util::Vec3& eyePos)
        {
            //VSetViewTransform(mat, invMat, eyePos);

            _ViewMatrixBuffer buffer;
            buffer.m_view = mat.m_m;
            buffer.m_invView = invMat.m_m;
            buffer.m_eyePos.x = eyePos.m_v.x;
            buffer.m_eyePos.y = eyePos.m_v.y;
            buffer.m_eyePos.z = eyePos.m_v.z;
            buffer.m_eyePos.w = 0;

            m_viewMatrixStack.Push(buffer);
            _ViewMatrixBuffer* pBuffer = (_ViewMatrixBuffer*)m_constBuffer[eViewBuffer]->VMap();
            *pBuffer = buffer;
            m_constBuffer[eViewBuffer]->VUnmap();

            chimera::CmGetApp()->VGetHumanView()->VGetSceneGraph()->VResetVisibility();
        }

        VOID Renderer::VPopViewTransform(VOID)
        {
            if(m_viewMatrixStack.Size() <= 1) 
            {
                return;
            }
            m_viewMatrixStack.Pop();
            _ViewMatrixBuffer buffer = m_viewMatrixStack.Peek();

            _ViewMatrixBuffer* pBuffer = (_ViewMatrixBuffer*)m_constBuffer[eViewBuffer]->VMap();
            *pBuffer = buffer;
            m_constBuffer[eViewBuffer]->VUnmap();

            chimera::CmGetApp()->VGetHumanView()->VGetSceneGraph()->VResetVisibility();
        }

        VOID Renderer::VPushMaterial(chimera::IMaterial& mat) 
        {
            _MaterialBuffer* mb = (_MaterialBuffer*)m_constBuffer[eMaterialBuffer]->VMap();
            mb->m_ambient = mat.VGetAmbient().m_v;
            mb->m_specular = mat.VGetSpecular().m_v;
            mb->m_diffuse = mat.VGetDiffuse().m_v;
            mb->m_specularExpo = mat.VGetSpecularExpo();
            mb->m_illum = mat.VGetReflectance();
            mb->m_textureSCale = mat.VGetTextureScale();
            m_constBuffer[eMaterialBuffer]->VUnmap();
        }

        VOID Renderer::VPopMaterial(VOID)
        {

        }

        VOID Renderer::VSetProjectionTransform(CONST util::Mat4& mat, FLOAT distance) 
        {
            m_projectionMatrixStack.Clear();
            _ProjectionMatrixBuffer buffer;
            buffer.m_projection = mat.m_m;
            buffer.m_viewDistance.x = buffer.m_viewDistance.y = buffer.m_viewDistance.z = buffer.m_viewDistance.w = distance;
            m_projectionMatrixStack.Push(buffer);
            _ProjectionMatrixBuffer* pBuffer = (_ProjectionMatrixBuffer*)m_constBuffer[eProjectionBuffer]->VMap();
            *pBuffer = buffer;
            m_constBuffer[eProjectionBuffer]->VUnmap();

            chimera::CmGetApp()->VGetHumanView()->VGetSceneGraph()->VResetVisibility();
        }

        VOID Renderer::VPushProjectionTransform(CONST util::Mat4& mat, FLOAT distance) 
        {
            _ProjectionMatrixBuffer buffer;
            buffer.m_projection = mat.m_m;
            buffer.m_viewDistance.x = buffer.m_viewDistance.y = buffer.m_viewDistance.z = buffer.m_viewDistance.w = distance;
            m_projectionMatrixStack.Push(buffer);
            _ProjectionMatrixBuffer* pBuffer = (_ProjectionMatrixBuffer*)m_constBuffer[eProjectionBuffer]->VMap();
            *pBuffer = buffer;
            m_constBuffer[eProjectionBuffer]->VUnmap();

            chimera::CmGetApp()->VGetHumanView()->VGetSceneGraph()->VResetVisibility();
        }

        VOID Renderer::VPopProjectionTransform(VOID)
        {
            if(m_projectionMatrixStack.Size() <= 1) 
            {
                return;
            }

            m_projectionMatrixStack.Pop();
            _ProjectionMatrixBuffer buffer = m_projectionMatrixStack.Peek();

            _ProjectionMatrixBuffer* pBuffer = (_ProjectionMatrixBuffer*)m_constBuffer[eProjectionBuffer]->VMap();
            *pBuffer = buffer;
            m_constBuffer[eProjectionBuffer]->VUnmap();

            chimera::CmGetApp()->VGetHumanView()->VGetSceneGraph()->VResetVisibility();
        }

        /*VOID D3DRenderer::SetActorId(UINT id)
        {
            chimera::ConstBuffer* buffer = GetBuffer(chimera::eActorIdBuffer);
            UINT* i = (UINT*)buffer->Map();
            i[0] = id;
            i[1] = i[2] = i[3] = 0;
            buffer->Unmap();
        } */

        VOID Renderer::SetPointLightShadowCubeMapSampler(ID3D11ShaderResourceView* view)
        {
            //SetSampler(ePointLightShadowCubeMapSampler, view);
        }

        VOID Renderer::SetCubeMapViews(CONST util::Mat4 mats[6])
        {
            _CubeMapViewsBuffer* mb = (_CubeMapViewsBuffer*)m_constBuffer[eCubeMapViewsBuffer]->VMap();
            for(UCHAR i = 0; i < 6; ++i)
            {
                mb->m_views[i] = mats[i].m_m;
            }
            m_constBuffer[eCubeMapViewsBuffer]->VUnmap();
        }

        VOID Renderer::SetLightSettings(CONST util::Vec4& color, CONST util::Vec3& position, FLOAT radius)
        {
            SetLightSettings(color, position, position, radius, 0.f, 1.f);
        }

        VOID Renderer::SetLightSettings(CONST util::Vec4& color, CONST util::Vec3& position, CONST util::Vec3& viewDir, FLOAT radius, FLOAT angel, FLOAT intensity)
        {
            _LightSettingsBuffer* plb = (_LightSettingsBuffer*)m_constBuffer[eLightingBuffer]->VMap();
            plb->m_colorNRadiusW.x = color.x;
            plb->m_colorNRadiusW.y = color.y;
            plb->m_colorNRadiusW.z = color.z;
            plb->m_colorNRadiusW.w = radius;
            plb->m_position.x = position.x;
            plb->m_position.y = position.y;
            plb->m_position.z = position.z;
            plb->m_position.w = intensity;
            plb->m_viewDirNAngel.x = viewDir.x;
            plb->m_viewDirNAngel.y = viewDir.y;
            plb->m_viewDirNAngel.z = viewDir.z;
            plb->m_viewDirNAngel.w = angel;
            m_constBuffer[eLightingBuffer]->VUnmap();
        }

        VOID Renderer::SetCSMSettings(CONST util::Mat4& view, CONST util::Mat4& iView, CONST util::Mat4 projections[3], CONST util::Vec3& lightPos, CONST FLOAT distances[3])
        {
            _LightingBuffer* lb = (_LightingBuffer*)m_constBuffer[eEnvLightingBuffer]->VMap();
            lb->m_view = view.m_m;
            lb->m_iView = iView.m_m;
            lb->m_projection[0] = projections[0].m_m;
            lb->m_projection[1] = projections[1].m_m;
            lb->m_projection[2] = projections[2].m_m;
            lb->m_lightPos.x = lightPos.x;
            lb->m_lightPos.y = lightPos.y;
            lb->m_lightPos.z = lightPos.z;
            lb->m_distances.x = distances[0];
            lb->m_distances.y = distances[1];
            lb->m_distances.z = distances[2];
            m_constBuffer[eEnvLightingBuffer]->VUnmap();
        }

        chimera::IConstShaderBuffer* Renderer::VGetConstShaderBuffer(ConstShaderBufferSlot slot)
        {
            return m_constBuffer[slot];
        }

        VOID Renderer::VSetViewPort(UINT w, UINT h)
        {
            D3D11_VIEWPORT viewPort;
            viewPort.Height = (FLOAT)h;
            viewPort.Width = (FLOAT)w;
            viewPort.MinDepth = 0;
            viewPort.MaxDepth = 1;
            viewPort.TopLeftX = 0;
            viewPort.TopLeftY = 0;
            chimera::d3d::GetContext()->RSSetViewports(1, &viewPort);
        }

        VOID Renderer::VPushDepthStencilState(IDepthStencilState* dsState, UINT stencilRef)
        {
            m_depthStencilStateStack.Push(dsState);
            chimera::d3d::GetContext()->OMSetDepthStencilState((ID3D11DepthStencilState*)dsState->VGetDevicePtr(), stencilRef);
        }
        
        VOID Renderer::VPopDepthStencilState(VOID)
        {
            if(m_depthStencilStateStack.Size() > 1)
            {
                m_depthStencilStateStack.Pop();
                chimera::d3d::GetContext()->OMSetDepthStencilState((ID3D11DepthStencilState*)m_depthStencilStateStack.Peek()->VGetDevicePtr(), 0);
            }
        }

        VOID Renderer::VPushRasterState(IRasterState* state)
        {
            m_rasterStateStack.Push(state);
            chimera::d3d::GetContext()->RSSetState((ID3D11RasterizerState*)state->VGetDevicePtr());
        }

        VOID Renderer::VPopRasterState(VOID)
        {
            if(m_rasterStateStack.Size() > 1)
            {
                m_rasterStateStack.Pop();
                chimera::d3d::GetContext()->RSSetState((ID3D11RasterizerState*)m_rasterStateStack.Peek()->VGetDevicePtr());
            }
        }

        VOID Renderer::VPushBlendState(IBlendState* state)
        {
            m_blendStateStack.Push(state);
            chimera::d3d::GetContext()->OMSetBlendState((ID3D11BlendState*)state->VGetDevicePtr(), NULL, -1);
        }

        VOID Renderer::VPopBlendState(VOID)
        {
            if(m_blendStateStack.Size() > 1)
            {
                m_blendStateStack.Pop();
                chimera::d3d::GetContext()->OMSetBlendState((ID3D11BlendState*)m_blendStateStack.Peek()->VGetDevicePtr(), NULL, -1);
            }
        }

        VOID Renderer::VSetDiffuseTexture(IDeviceTexture* texture)
        {
            VSetTexture(eDiffuseColorSampler, texture);
        }

        VOID Renderer::VSetTexture(TextureSlot slot, IDeviceTexture* texture)
        {
            ID3D11ShaderResourceView* v = (ID3D11ShaderResourceView*)(texture == NULL ? NULL : texture->VGetViewDevicePtr());
            if(m_currentSetSampler[slot] != v)
            {
                chimera::d3d::GetContext()->PSSetShaderResources(slot, 1, &v);
                m_currentSetSampler[slot] = v;
            }
        }

        VOID Renderer::VSetTextures(TextureSlot slot, IDeviceTexture** texture, UINT count)
        {
            static ID3D11ShaderResourceView* v[D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT];
            
            assert(count <= D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT);

            TBD_FOR_INT(count) { v[i] = (ID3D11ShaderResourceView*)(texture[i] == NULL ? NULL : texture[i]->VGetViewDevicePtr()); }
            SetSampler(slot, v, count, PS_Stage);
        }

        VOID Renderer::SetSampler(TextureSlot startSlot, ID3D11ShaderResourceView** view, UINT count, UINT stages)
        {
            for(UINT i = 0; i < count; ++i)
            {
                if(m_currentSetSampler[startSlot + i] != view[i])
                {
                    /*
                    if(stages & VS_Stage)
                    {
                       d3d::GetContext()->VSSetShaderResources(slot, 1, &view);
                    } */
                    if(stages & PS_Stage)
                    {
                       chimera::d3d::GetContext()->PSSetShaderResources(startSlot + i, 1, view+i);
                    }
                    /*
                    if(stages & GS_STAGE)
                    {
                        d3d::GetContext()->GSSetShaderResources(slot, 1, &view);
                    } */
                    m_currentSetSampler[startSlot + i] = view[i];
                }
            }

        }

        IAlbedoBuffer* Renderer::VGetAlbedoBuffer(VOID)
        {
            return m_pDefShader;
        }

        CM_HWND Renderer::VGetWindowHandle(VOID)
        {
            return (CM_HWND)chimera::d3d::g_hWnd;
        }

        VOID Renderer::VDrawScreenQuad(INT x, INT y, INT w, INT h)
        {
            m_screenQuadProgram->VBind();
            FLOAT _x = -1.0f + 2 * x / (FLOAT)VGetWidth();
            FLOAT _y = -1.0f + 2 * y / (FLOAT)VGetHeight();
            FLOAT _w = _x + 2 * w / (FLOAT)VGetWidth();
            FLOAT _h = _y + 2 * h / (FLOAT)VGetHeight();
            FLOAT localVertices[20] = 
            {
                _x, _y, 0, 0, 1,
                _w, _y, 0, 1, 1,
                _x, _h, 0, 0, 0,
                _w, _h, 0, 1, 0,
            };

            IGeometry* quad = geometryfactroy::GetGlobalScreenQuadCPU();
            quad->VGetVertexBuffer()->VSetData(localVertices, sizeof(FLOAT) * 20);
            quad->VBind();
            quad->VDraw();
        }

        VOID Renderer::VDrawLine(INT x, INT y, INT w, INT h)
        {
            FLOAT _x = -1.0f + 2 * x / (FLOAT)VGetWidth();
            FLOAT _y = -1.0f + 2 * y / (FLOAT)VGetHeight();
            FLOAT _w = _x + 2 * w / (FLOAT)VGetWidth();
            FLOAT _h = _y + 2 * h / (FLOAT)VGetHeight();
            FLOAT localVertices[10] = 
            {
                _x, _y, 0, 0, 0,
                _w, _h, 0, 1, 1,
            };

            IGeometry* line = geometryfactroy::GetGlobalLineCPU();
            line->VGetVertexBuffer()->VSetData(localVertices, sizeof(FLOAT) * 10);
            line->VBind();
            line->VDraw();
        }

        VOID Renderer::VDrawScreenQuad(VOID)
        {
            geometryfactroy::GetGlobalScreenQuad()->VBind();
            geometryfactroy::GetGlobalScreenQuad()->VDraw();
        }

        VOID Renderer::Delete(VOID)
        {
            for(UCHAR i = 0; i < BufferCnt; ++i)
            {
                SAFE_DELETE(m_constBuffer[i]);
            }
        }

        VOID Renderer::VPushAlphaBlendState(VOID)
        {
            VPushBlendState(m_pAlphaBlendState);
        }

        Renderer::~Renderer(VOID) 
        {
            VDestroy();
        }

        //Defshader
        AlbedoBuffer::AlbedoBuffer(UINT w, UINT h)
        {
            for(UINT i = 0; i < Diff_SamplersCnt; ++i)
            {
                m_targets[i] = new RenderTarget();
            }
            VOnRestore(w, h);
        }

        VOID AlbedoBuffer::VOnRestore(UINT w, UINT h)
        {
            m_height = h;
            m_width = w;
            /*for(UINT i = 0; i < Diff_SamplersCnt; ++i)
            {
                SAFE_DELETE(m_targets[i]);
                m_targets[i] = new d3d::RenderTarget;
            }*/
            m_targets[eDiff_WorldPositionTarget]->VOnRestore(m_width, m_height, eFormat_R32G32B32A32_FLOAT, TRUE);
            m_targets[eDiff_NormalsTarget]->VOnRestore(m_width, m_height, eFormat_R32G32B32A32_FLOAT, FALSE);
            m_targets[eDiff_DiffuseMaterialSpecRTarget]->VOnRestore(m_width, m_height, eFormat_R16G16B16A16_FLOAT, FALSE);
            m_targets[eDiff_AmbientMaterialSpecGTarget]->VOnRestore(m_width, m_height, eFormat_R16G16B16A16_FLOAT, FALSE);
            m_targets[eDiff_DiffuseColorSpecBTarget]->VOnRestore(m_width, m_height, eFormat_R16G16B16A16_FLOAT, FALSE);
            m_targets[eDiff_ReflectionStrTarget]->VOnRestore(m_width, m_height, eFormat_R16_FLOAT, FALSE);

            for(UINT i = 0; i < Diff_SamplersCnt; ++i)
            {
                m_views[i] = m_targets[i]->GetRenderTargetView();
            }

            m_viewPort.MinDepth = 0;
            m_viewPort.MaxDepth = 1;
            m_viewPort.TopLeftX = 0;
            m_viewPort.TopLeftY = 0;

            m_viewPort.Width = (FLOAT)m_width;
            m_viewPort.Height = (FLOAT)m_height;
        }

        IRenderTarget* AlbedoBuffer::VGetDepthStencilTarget(VOID)
        {
            return m_targets[0];//->GetDepthStencilView();
        }

        VOID AlbedoBuffer::VClearAndBindRenderTargets(VOID)
        {
            for(UINT i = 0; i < Diff_SamplersCnt; ++i)
            {
                m_targets[i]->VClear();
            }

            chimera::d3d::GetContext()->RSSetViewports(1, &m_viewPort);

            chimera::d3d::GetContext()->OMSetRenderTargets(Diff_SamplersCnt, m_views, m_targets[0]->GetDepthStencilView());
        }

        
        VOID AlbedoBuffer::VUnbindRenderTargets(VOID)
        {
            ID3D11RenderTargetView* v[Diff_SamplersCnt] = {NULL, NULL, NULL, NULL, NULL, NULL};
            chimera::d3d::GetContext()->OMSetRenderTargets(Diff_SamplersCnt, v, NULL);
        }

        chimera::IRenderTarget* AlbedoBuffer::VGetRenderTarget(Diff_RenderTarget stage)
        {
            return m_targets[stage];
        }

        AlbedoBuffer::~AlbedoBuffer(VOID)
        {
            for(UINT i = 0; i < Diff_SamplersCnt; ++i)
            {
                SAFE_DELETE(m_targets[i]);
            }
        }
    }
}