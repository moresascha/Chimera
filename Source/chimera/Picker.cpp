#include "Picker.h"
#include "Event.h"
#include "D3DGraphics.h"

namespace chimera
{

    ActorPicker::ActorPicker(void)  : m_currentActor(CM_INVALID_ACTOR_ID), m_created(FALSE), m_pTexture(NULL), m_pShaderProgram(NULL), m_pRenderTarget(NULL)
    {
        ADD_EVENT_LISTENER(this, &ActorPicker::PickActorDelegate, CM_EVENT_PICK_ACTOR);
        ADD_EVENT_LISTENER(this, &ActorPicker::ActorDeletedDelegate, CM_EVENT_ACTOR_DELETED);
    }

    ActorId ActorPicker::VPick(void)
    {
        Render();
        PostRender();
        return m_currentActor;
    }

    bool ActorPicker::VHasPicked(void) const
    {
        return m_currentActor != CM_INVALID_ACTOR_ID;
    }

    void ActorPicker::PostRender(void)
    {
        //todo: move to renderer
        D3D11_MAPPED_SUBRESOURCE res;
        ID3D11DeviceContext* ctx = (ID3D11DeviceContext*)CmGetApp()->VGetRenderer()->VGetContext();
        ID3D11Resource* dst = (ID3D11Resource*)m_pTexture->VGetDevicePtr();
        ID3D11Resource* src = (ID3D11Resource*)m_pRenderTarget->VGetTexture()->VGetDevicePtr();

        ctx->CopyResource(dst, src);
        D3D_SAVE_CALL(ctx->Map(dst, 0, D3D11_MAP_READ, 0, &res));
        m_currentActor = ((uint*)(res.pData))[0];
        ctx->Unmap(dst, 0);

        IConstShaderBuffer* buffer = CmGetApp()->VGetHumanView()->VGetRenderer()->VGetConstShaderBuffer(chimera::eSelectedActorIdBuffer);
        uint* b = (uint*)buffer->VMap();
        b[0] = m_currentActor;
        buffer->VUnmap();
    }

    void ActorPicker::PickActorDelegate(IEventPtr ptr)
    {
        std::shared_ptr<PickActorEvent> event = std::static_pointer_cast<PickActorEvent>(ptr);

        VPick();

        event->CallCallBack(m_currentActor);
    }

    void ActorPicker::ActorDeletedDelegate(IEventPtr ptr)
    {
        std::shared_ptr<ActorDeletedEvent> event = std::static_pointer_cast<ActorDeletedEvent>(ptr);
        if(event->m_id == m_currentActor)
        {
            m_currentActor = CM_INVALID_ACTOR_ID;
        }
    }

    void ActorPicker::Render(void)
    {
        m_pRenderTarget->VBind();
        m_pRenderTarget->VClear();
        CmGetApp()->VGetHumanView()->VGetRenderer()->VPushProjectionTransform(m_projection, 1000.0f);
        m_pShaderProgram->VBind();
        CmGetApp()->VGetHumanView()->VGetSceneGraph()->VOnRender(CM_RENDERPATH_PICK);
        CmGetApp()->VGetHumanView()->VGetRenderer()->VPopProjectionTransform();
    }

    bool ActorPicker::VOnRestore(void)
    {
        if(m_created)
        {
            return true;
        }

        m_created = true;

        CMShaderProgramDescription shaderDesc;
        shaderDesc.fs.file = L"Picking.hlsl";
        shaderDesc.vs.file = L"Picking.hlsl";

        shaderDesc.vs.function = "Picking_VS";
        shaderDesc.fs.function = "Picking_PS";

        shaderDesc.vs.layoutCount = 3;

        shaderDesc.vs.inputLayout[0].instanced = false;
        shaderDesc.vs.inputLayout[0].name = "POSITION";
        shaderDesc.vs.inputLayout[0].position = 0;
        shaderDesc.vs.inputLayout[0].slot = 0;
        shaderDesc.vs.inputLayout[0].format = eFormat_R32G32B32_FLOAT;

        shaderDesc.vs.inputLayout[1].instanced = false;
        shaderDesc.vs.inputLayout[1].name = "NORMAL";
        shaderDesc.vs.inputLayout[1].position = 1;
        shaderDesc.vs.inputLayout[1].slot = 0;
        shaderDesc.vs.inputLayout[1].format = eFormat_R32G32B32_FLOAT;

        shaderDesc.vs.inputLayout[2].instanced = false;
        shaderDesc.vs.inputLayout[2].name = "TEXCOORD";
        shaderDesc.vs.inputLayout[2].position = 2;
        shaderDesc.vs.inputLayout[2].slot = 0;
        shaderDesc.vs.inputLayout[2].format = eFormat_R32G32_FLOAT;

        m_pShaderProgram = CmGetApp()->VGetHumanView()->VGetRenderer()->VGetShaderCache()->VCreateShaderProgram("Picker", &shaderDesc);

        /*m_pRenderTarget->SetUsage(D3D11_USAGE_DEFAULT);
        m_pRenderTarget->SetBindflags(D3D11_BIND_RENDER_TARGET);
        m_renderTarget->SetClearColor(-1, -1, -1, -1);
        m_renderTarget->SetCPUAccess(D3D11_CPU_ACCESS_READ);
        
        uint w = 1, h = 1;
        m_pTexture = new chimera::D3DTexture2D();
        m_pTexture->SetWidth(w);
        m_pTexture->SetHeight(h);
        m_pTexture->SetFormat(DXGI_FORMAT_R32_UINT);
        m_pTexture->SetUsage(D3D11_USAGE_STAGING);
        m_pTexture->SetCPUAccess(D3D11_CPU_ACCESS_READ);
        */

        CMTextureDescription texDesc;
        ZeroMemory(&texDesc, sizeof(CMTextureDescription));

        texDesc.width = 1;
        texDesc.height = 1;
        texDesc.cpuAccess = D3D11_CPU_ACCESS_READ;
        texDesc.format = eFormat_R32_UINT;
        texDesc.usage = D3D11_USAGE_STAGING;

        m_pTexture = CmGetApp()->VGetHumanView()->VGetGraphicsFactory()->VCreateTexture(&texDesc).release();
        
        if(!m_pTexture->VCreate())
        {
            return false;
        }

        m_pRenderTarget = CmGetApp()->VGetHumanView()->VGetGraphicsFactory()->VCreateRenderTarget().release();
        if(!m_pRenderTarget->VOnRestore(texDesc.width, texDesc.height, eFormat_R32_UINT))
        {
            return false;
        }

        m_projection = util::Mat4::CreatePerspectiveLH(XM_PIDIV2, 1, 0.01f, 1000.0f);

        return true;
    }

    IRenderTarget* ActorPicker::GetTarget(void)
    {
        return m_pRenderTarget;
    }

    ActorPicker::~ActorPicker(void)
    {
        REMOVE_EVENT_LISTENER(this, &ActorPicker::PickActorDelegate, CM_EVENT_PICK_ACTOR);
        REMOVE_EVENT_LISTENER(this, &ActorPicker::ActorDeletedDelegate, CM_EVENT_ACTOR_DELETED);
        if(m_pTexture)
        {
            m_pTexture->VDestroy();
        }
        SAFE_DELETE(m_pRenderTarget);
        SAFE_DELETE(m_pTexture);
    }
}
