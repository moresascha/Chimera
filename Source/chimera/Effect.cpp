#include "Effect.h"
#include "util.h"
#include "Vec3.h"

namespace chimera
{
    void DefaultDraw(void)
    {
        CmGetApp()->VGetHumanView()->VGetRenderer()->VDrawScreenQuad();
    }

    class DefaultParams : public IEffectParmaters
    {
    public:
        void VApply(void) {}

        void VOnRestore(void) {}

        ~DefaultParams(void) {}
    };

    Effect::Effect(void) 
        : m_pPixelShader(NULL), m_target(NULL), m_w(0), m_h(0),
        m_pParams(NULL), m_isProcessed(false), m_srcCount(0)
    {
        m_pfDraw = DefaultDraw;
        memset(m_source, NULL, sizeof(IEffect*) * 4);

        m_pParams = new DefaultParams();
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

        m_pParams->VOnRestore();

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
        VAddSource(e->VGetTarget());
    }

    IEffectParmaters* Effect::VSetParameters(std::unique_ptr<IEffectParmaters>& params)
    {
        SAFE_DELETE(m_pParams);
        m_pParams = params.release();
        m_pParams->VOnRestore();
        return m_pParams;
    }

    void Effect::VAddSource(IDeviceTexture* src)
    {
        //m_source[m_srcCount++] = src;
        assert(m_srcCount < 17);
    }

    void Effect::VAddSource(IRenderTarget* src)
    {
        //VAddSource(src->VGetTexture());
        m_source[m_srcCount++] = src;
    }

    IRenderTarget* Effect::VGetTarget(void)
    {
        return m_target;
    }

    IRenderTarget* Effect::VGetSource(uint index)
    {
        assert(index < 16);
        return (IRenderTarget*)m_source[index];
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

        for(auto& it = m_requirements.begin(); it != m_requirements.end(); ++it)
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

        const uint c_startSlot = eEffect0;
        uint startSlot = c_startSlot;
        IDeviceTexture* view = NULL;

        for(int i = 0; i < m_srcCount; ++i)
        {
            view = m_source[i]->VGetTexture();
            CmGetApp()->VGetHumanView()->VGetRenderer()->VSetTexture((TextureSlot)startSlot++, view);
        }
// 
//         for(auto& it = m_requirements.begin(); it != m_requirements.end(); ++it)
//         {
//             IEffect* e = (*it);
//             view = e->VGetTarget()->VGetTexture();
//             CmGetApp()->VGetHumanView()->VGetRenderer()->VSetTexture((TextureSlot)startSlot++, view);
//         }
        
        m_pParams->VApply();

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
        SAFE_DELETE(m_pParams);
    }

    EffectChain::EffectChain(IEffectFactory* factory) : m_w(0), m_h(0), m_pEffectFactory(factory), m_pVertexShader(NULL), m_leaf(NULL)
    {
    }

    IRenderTarget* EffectChain::VGetResult(void)
    {
        return m_leaf->VGetSource();
    }

    IEffect* EffectChain::VAppendEffect(std::unique_ptr<IEffect>& effect, float percentofw, float percentofh)
    {
        ErrorLog log;
        if(!effect->VOnRestore(CmGetApp()->VGetWindowWidth(), CmGetApp()->VGetWindowHeight(), &log))
        {
            LOG_CRITICAL_ERROR(log.c_str());
        }

        if(m_leaf)
        {
            effect->VAddRequirement(m_leaf);
            m_pTarget = m_leaf->VGetTarget();
        }

        m_leaf = effect.get();

        m_effects.push_back(std::move(effect));

        return m_leaf;
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

        for(auto& it = m_effects.begin(); it != m_effects.end(); ++it)
        {
            (*it)->VReset();
        }
    }

    /*
    SSAA::SSAA(void) : m_pEffect(NULL), m_pKernelTexture(NULL), m_pNoiseTexture(NULL)
    {

    }

    void SSAA::VCreate(const CMShaderDescription& shaderDesc, float w, float h)
    {
        SAFE_DELETE(m_pEffect);
        m_pEffect = new Effect();
        m_pEffect->VCreate(shaderDesc, w, h);
    }

    void SSAA::VSetParameters(IEffectParmaters* params)
    {
        m_pEffect->VSetParameters(params);
    }

    void SSAA::VSetDrawMethod(EffectDrawMethod dm)
    {
        m_pEffect->VSetDrawMethod(dm);
    }

    void SSAA::VAddRequirement(IEffect* e)
    {
        m_pEffect->VAddRequirement(e);
    }

    void SSAA::VReset(void)
    {
        m_pEffect->VReset();
    }

    IRenderTarget* SSAA::VGetSource(uint index)
    {
        return m_pEffect->VGetSource(index);
    }

    void SSAA::VAddSource(IRenderTarget* src)
    {
        m_pEffect->VAddSource(src);
    }

    void SSAA::VAddSource(IDeviceTexture* src)
    {
        m_pEffect->VAddSource(src);
    }

    float2 SSAA::VGetViewPort(void)
    {
        return m_pEffect->VGetViewPort();
    }

    void SSAA::VProcess(void)
    {
        m_pEffect->VProcess();
    }

    void SSAA::VSetTarget(IRenderTarget* target)
    {
        m_pEffect->VSetTarget(target);
    }

    void SSAA::VSetTarget(std::unique_ptr<IRenderTarget> target)
    {
        m_pEffect->VSetTarget(std::move(target));
    }

    IRenderTarget* SSAA::VGetTarget(void)
    {
        return m_pEffect->VGetTarget();
    }*/

    float Lerp(float a, float b, float t)
    {
        return a + t*(b-a);
    }

    SSAAParameters::SSAAParameters(void) : m_pNoiseTexture(NULL), m_pKernelTexture(NULL)
    {

    }

    void SSAAParameters::VApply(void)
    {
        CmGetApp()->VGetHumanView()->VGetRenderer()->VSetTexture(eEffect1, m_pKernelTexture);
        CmGetApp()->VGetHumanView()->VGetRenderer()->VSetTexture(eEffect2, m_pNoiseTexture);
    }

    void SSAAParameters::VOnRestore(void)
    {
        CMTextureDescription desc;

        if(m_pKernelTexture)
        {
            m_pKernelTexture->VDestroy();
            m_pNoiseTexture->VDestroy();
        }

        SAFE_DELETE(m_pKernelTexture);
        SAFE_DELETE(m_pNoiseTexture);

        util::cmRNG rng;

        const uint kernelSize = 64;
        const uint noiseSize = 16;

        XMFLOAT3 sampleData[kernelSize];
        XMFLOAT2 noiseData[noiseSize];

        for(int i = 0; i < kernelSize; ++i)
        {
            util::Vec3 s;
            s.x = rng.NextCubeFloat();
            s.y = rng.NextCubeFloat();
            s.z = rng.NextFloat();
            s.Normalize();
            float scale = (float)i / (float)kernelSize;
            scale = Lerp(0.1f, 1.0f, scale * scale);
            s.Scale(scale);

            sampleData[i].x = s.x;
            sampleData[i].y = s.y;
            sampleData[i].z = s.z;
        }

        for(int i = 0; i < noiseSize; ++i)
        {
            util::Vec3 s;
            s.x = rng.NextCubeFloat();
            s.y = rng.NextCubeFloat();
            s.z = 0;
            s.Normalize();

            noiseData[i].x = s.x;
            noiseData[i].y = s.y;
        }
        
        ZeroMemory(&desc, sizeof(CMTextureDescription));
        desc.width = kernelSize;
        desc.height = 1;
        desc.data = sampleData;
        desc.miscflags = eTextureMiscFlags_BindShaderResource;
        desc.format = eFormat_R32G32B32_FLOAT;

        m_pKernelTexture = CmGetApp()->VGetHumanView()->VGetGraphicsFactory()->VCreateTexture(&desc).release();
        m_pKernelTexture->VCreate();

        desc.width = 4;
        desc.height = 4;
        desc.data = noiseData;
        desc.format = eFormat_R32G32_FLOAT;

        m_pNoiseTexture = CmGetApp()->VGetHumanView()->VGetGraphicsFactory()->VCreateTexture(&desc).release();
        m_pNoiseTexture->VCreate();
    }

    SSAAParameters::~SSAAParameters(void)
    {
        if(m_pNoiseTexture)
        {
            m_pKernelTexture->VDestroy();
            m_pNoiseTexture->VDestroy();
        }
        SAFE_DELETE(m_pKernelTexture);
        SAFE_DELETE(m_pNoiseTexture);
    }
}
