#include "Effect.h"

namespace chimera
{

    VOID DefaultDraw(VOID)
    {
        CmGetApp()->VGetHumanView()->VGetRenderer()->VDrawScreenQuad();
    }

    Effect::Effect(VOID) 
        : m_pPixelShader(NULL), m_target(NULL), m_source(NULL), m_w(0), m_h(0),
        m_params(NULL), m_isProcessed(FALSE)
    {
        m_pfDraw = DefaultDraw;
    }

    VOID Effect::VCreate(CONST CMShaderDescription& shaderDesc, FLOAT w, FLOAT h)
    {
        m_shaderDesc = shaderDesc;
        m_w = w;
        m_h = h;
    }

    VOID Effect::VSetDrawMethod(EffectDrawMethod dm)
    {
        m_pfDraw = dm;
    }

    BOOL Effect::VOnRestore(UINT w, UINT h, ErrorLog* log)
    {
        if(!m_pPixelShader)
        {
            m_pPixelShader = CmGetApp()->VGetHumanView()->VGetRenderer()->VGetShaderCache()->VCreateShader(m_shaderDesc.function, &m_shaderDesc, eShaderType_FragmentShader);
        }

        if(m_target)
        {
            m_target->VOnRestore(max(1, (UINT)(w * m_w)), max(1, (UINT)(h * m_h)), eFormat_R32G32B32A32_FLOAT, FALSE);
        }

        return TRUE;
    }

    FLOAT2 Effect::VGetViewPort(VOID)
    {
        FLOAT2 vp;
        vp.x = m_w;
        vp.y = m_h;
        return vp;
    }

    VOID Effect::VAddRequirement(IEffect* e)
    {
        std::unique_ptr<IRenderTarget> t(CmGetApp()->VGetHumanView()->VGetGraphicsFactory()->VCreateRenderTarget());
        e->VSetTarget(std::move(t));
        FLOAT2 vp = e->VGetViewPort();
        e->VGetTarget()->VOnRestore((UINT)(vp.x * (FLOAT)CmGetApp()->VGetWindowWidth()), (UINT)(vp.y * (FLOAT)CmGetApp()->VGetWindowHeight()), eFormat_R32G32B32A32_FLOAT, FALSE);
        m_requirements.push_back(e);
    }

    VOID Effect::VSetParameters(IEffectParmaters* params)
    {
        m_params = params;
    }

    VOID Effect::VSetSource(IRenderTarget* src)
    {
        m_source = src;
    }

    IRenderTarget* Effect::VGetTarget(VOID)
    {
        return m_target;
    }

    VOID Effect::VSetTarget(IRenderTarget* target)
    {
        m_target = target;
    }

    VOID Effect::VSetTarget(std::unique_ptr<IRenderTarget> target)
    {
        m_ownedTarget = std::move(target);
        m_target = m_ownedTarget.get();
    }

    VOID Effect::VProcess(VOID)
    {
        if(m_isProcessed)
        {
            return;
        }

        for(auto it = m_requirements.begin(); it != m_requirements.end(); ++it)
        {
            (*it)->VProcess();
        }

        m_pPixelShader->VBind();

        if(m_target == NULL)
        {
            //CmGetApp()->VSetViewPort(0, 0, 1, 1);
            CmGetApp()->VGetHumanView()->VGetRenderer()->VClearAndBindBackBuffer();
        }
        else
        {
            m_target->VClear();
            m_target->VBind();
        }

        //m_params->VApply();

        CONST UINT c_startSlot = eEffect0;
        UINT startSlot = c_startSlot;
        IDeviceTexture* view = NULL;

        if(m_source)
        {
            view = m_source->VGetTexture();
            CmGetApp()->VGetHumanView()->VGetRenderer()->VSetTexture((TextureSlot)startSlot++, view);
        }

        for(auto it = m_requirements.begin(); it != m_requirements.end(); ++it)
        {
            IEffect* e = (*it);
            view = e->VGetTarget()->VGetTexture();
            CmGetApp()->VGetHumanView()->VGetRenderer()->VSetTexture((TextureSlot)startSlot++, view);
        }

        m_pfDraw();

        view = NULL;

        for(INT i = startSlot-1; i >= c_startSlot; --i)
        {
            CmGetApp()->VGetHumanView()->VGetRenderer()->VSetTexture((TextureSlot)i, view);
        }

        m_isProcessed = TRUE;
    }

    VOID Effect::VReset(VOID)
    {
        m_isProcessed = FALSE;
    }

    Effect::~Effect(VOID)
    {
    }

    EffectChain::EffectChain(IEffectFactory* factory) : m_w(0), m_h(0), m_pEffectFactory(factory), m_pVertexShader(NULL)
    {
    }

    IRenderTarget* EffectChain::VGetResult(VOID)
    {
        return m_pTarget;
    }

    IEffect* EffectChain::VCreateEffect(CONST CMShaderDescription& shaderDesc, FLOAT w, FLOAT h)
    {
        std::unique_ptr<IEffect> e(m_pEffectFactory->VCreateEffect());

        e->VCreate(shaderDesc, w, h);

        ErrorLog log;
        if(!e->VOnRestore(CmGetApp()->VGetWindowWidth(), CmGetApp()->VGetWindowHeight(), &log))
        {
            LOG_CRITICAL_ERROR(log.c_str());
        }

        m_leaf = e.get();

        m_effects.push_back(std::move(e));

        return m_leaf;
    }

    VOID EffectChain::VOnRestore(UINT w, UINT h)
    {
        if(!m_pVertexShader)
        {
            CMVertexShaderDescription desc;
            desc.layoutCount = 2;
            
            desc.inputLayout[0].format = eFormat_R32G32B32_FLOAT;
            desc.inputLayout[0].instanced = FALSE;
            desc.inputLayout[0].name = "POSITION";
            desc.inputLayout[0].position = 0;
            desc.inputLayout[0].slot = 0;

            desc.inputLayout[1].format = eFormat_R32G32_FLOAT;
            desc.inputLayout[1].instanced = FALSE;
            desc.inputLayout[1].name = "TEXCOORD";
            desc.inputLayout[1].position = 1;
            desc.inputLayout[1].slot = 0;

            desc.file = L"Effects.hlsl";
            desc.function = "Effect_VS";
            m_pVertexShader = CmGetApp()->VGetHumanView()->VGetRenderer()->VGetShaderCache()->VCreateShader(desc.function, &desc, eShaderType_VertexShader);
        }

        for(auto it = m_effects.begin(); it != m_effects.end(); ++it)
        {
            IEffect* effect = it->get();
            effect->VOnRestore(w, h);
        }
    }


    VOID EffectChain::VSetSource(IRenderTarget* src)
    {
        m_pSrc = src;
    }

    VOID EffectChain::VSetTarget(IRenderTarget* target)
    {
        m_pTarget = target;
    }

    EffectChain::~EffectChain(VOID)
    {

    }

    VOID EffectChain::VProcess(VOID)
    {
        
        //ID3D11VertexShader* tmp;
        //d3d::GetContext()->VSGetShader(&tmp, NULL, 0);

        m_pVertexShader->VBind();

        m_leaf->VProcess();

        /*
        for(auto it = m_effects.begin(); it != m_effects.end(); ++it)
        {
            Effect* e = *it;
            for(auto it2 = e->m_requirements.begin(); it2 != e->m_requirements.end(); ++it2)
            {
                if(!e->m_isProcessed)
                {
                    e->Process();
                }
            }
            (*it)->Process();
        } */

        for(auto it = m_effects.begin(); it != m_effects.end(); ++it)
        {
            (*it)->VReset();
        }

        //d3d::GetContext()->VSSetShader(tmp, NULL, 0);
    }
}
