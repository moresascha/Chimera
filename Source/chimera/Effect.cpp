#include "Effect.h"

namespace chimera
{

    void DefaultDraw(void)
    {
        CmGetApp()->VGetHumanView()->VGetRenderer()->VDrawScreenQuad();
    }

    Effect::Effect(void) 
        : m_pPixelShader(NULL), m_target(NULL), m_source(NULL), m_w(0), m_h(0),
        m_params(NULL), m_isProcessed(false)
    {
        m_pfDraw = DefaultDraw;
    }

    void Effect::VCreate(const CMShaderDescription& shaderDesc, float w, float h)
    {
        m_shaderDesc = shaderDesc;
        m_w = w;
        m_h = h;
    }

    void Effect::VSetDrawMethod(EffectDrawMethod dm)
    {
        m_pfDraw = dm;
    }

    bool Effect::VOnRestore(uint w, uint h, ErrorLog* log)
    {
        if(!m_pPixelShader)
        {
            m_pPixelShader = CmGetApp()->VGetHumanView()->VGetRenderer()->VGetShaderCache()->VCreateShader(m_shaderDesc.function, &m_shaderDesc, eShaderType_FragmentShader);
        }

        if(m_target)
        {
            m_target->VOnRestore(max(1, (uint)(w * m_w)), max(1, (uint)(h * m_h)), eFormat_R32G32B32A32_FLOAT, false);
        }

        return true;
    }

    float2 Effect::VGetViewPort(void)
    {
        float2 vp;
        vp.x = m_w;
        vp.y = m_h;
        return vp;
    }

    void Effect::VAddRequirement(IEffect* e)
    {
        std::unique_ptr<IRenderTarget> t(CmGetApp()->VGetHumanView()->VGetGraphicsFactory()->VCreateRenderTarget());
        e->VSetTarget(std::move(t));
        float2 vp = e->VGetViewPort();
        e->VGetTarget()->VOnRestore((uint)(vp.x * (float)CmGetApp()->VGetWindowWidth()), (uint)(vp.y * (float)CmGetApp()->VGetWindowHeight()), eFormat_R32G32B32A32_FLOAT, false);
        m_requirements.push_back(e);
        VSetSource(e->VGetTarget());
    }

    void Effect::VSetParameters(IEffectParmaters* params)
    {
        m_params = params;
    }

    void Effect::VSetSource(IRenderTarget* src)
    {
        m_source = src;
    }

    IRenderTarget* Effect::VGetTarget(void)
    {
        return m_target;
    }

    IRenderTarget* Effect::VGetSource(void)
    {
        return m_source;
    }

    void Effect::VSetTarget(IRenderTarget* target)
    {
        m_target = target;
    }

    void Effect::VSetTarget(std::unique_ptr<IRenderTarget> target)
    {
        m_ownedTarget = std::move(target);
        m_target = m_ownedTarget.get();
    }

    void Effect::VProcess(void)
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

        const uint c_startSlot = eEffect0;
        uint startSlot = c_startSlot;
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

        for(int i = startSlot-1; i >= c_startSlot; --i)
        {
            CmGetApp()->VGetHumanView()->VGetRenderer()->VSetTexture((TextureSlot)i, view);
        }

        m_isProcessed = true;
    }

    void Effect::VReset(void)
    {
        m_isProcessed = false;
    }

    Effect::~Effect(void)
    {
    }

    EffectChain::EffectChain(IEffectFactory* factory) : m_w(0), m_h(0), m_pEffectFactory(factory), m_pVertexShader(NULL), m_leaf(NULL)
    {
    }

    IRenderTarget* EffectChain::VGetResult(void)
    {
        return m_leaf->VGetSource();
    }

    IEffect* EffectChain::VAppendEffect(const CMShaderDescription& shaderDesc, float w, float h)
    {
        std::unique_ptr<IEffect> e(m_pEffectFactory->VCreateEffect());

        e->VCreate(shaderDesc, w, h);

        ErrorLog log;
        if(!e->VOnRestore(CmGetApp()->VGetWindowWidth(), CmGetApp()->VGetWindowHeight(), &log))
        {
            LOG_CRITICAL_ERROR(log.c_str());
        }

        if(m_leaf)
        {
            e->VAddRequirement(m_leaf);
            m_pTarget = m_leaf->VGetTarget();
        }

        m_leaf = e.get();

        m_effects.push_back(std::move(e));

        return m_leaf;
    }

    void EffectChain::VOnRestore(uint w, uint h)
    {
        if(!m_pVertexShader)
        {
            CMVertexShaderDescription desc;
            desc.layoutCount = 2;
            
            desc.inputLayout[0].format = eFormat_R32G32B32_FLOAT;
            desc.inputLayout[0].instanced = false;
            desc.inputLayout[0].name = "POSITION";
            desc.inputLayout[0].position = 0;
            desc.inputLayout[0].slot = 0;

            desc.inputLayout[1].format = eFormat_R32G32_FLOAT;
            desc.inputLayout[1].instanced = false;
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


    void EffectChain::VSetSource(IRenderTarget* src)
    {
        m_pSrc = src;
    }

    void EffectChain::VSetTarget(IRenderTarget* target)
    {
        //m_pTarget = target;
    }

    EffectChain::~EffectChain(void)
    {

    }

    void EffectChain::VProcess(void)
    {
        m_pVertexShader->VBind();

        m_leaf->VProcess();

        for(auto it = m_effects.begin(); it != m_effects.end(); ++it)
        {
            (*it)->VReset();
        }
    }
}
