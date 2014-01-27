#pragma once
#include "stdafx.h"

namespace chimera
{
    class DefaultParams : public IEffectParmaters
    {
    public:
        void VApply(void) {  }
    };

    class Effect : public IEffect
    {
    private:
        IRenderTarget* m_source;
        IRenderTarget* m_target;
        std::unique_ptr<IRenderTarget> m_ownedTarget;
        bool m_created, m_isProcessed;
        IShader* m_pPixelShader;
        float m_w, m_h;
        IEffectParmaters* m_params; //TODO
        std::vector<IEffect*> m_requirements;
        CMShaderDescription m_shaderDesc;
        EffectDrawMethod m_pfDraw;

    public:
        Effect(void);

        void VCreate(const CMShaderDescription& shaderDesc, float w, float h);

        void VSetParameters(IEffectParmaters* params);

        void VSetDrawMethod(EffectDrawMethod dm);

        void VAddRequirement(IEffect* e);

        void VReset(void);

        IRenderTarget* VGetSource(void);

        void VSetSource(IRenderTarget* src);

        float2 VGetViewPort(void);

        void VProcess(void);

        void VSetTarget(IRenderTarget* target);

        void VSetTarget(std::unique_ptr<IRenderTarget> target);

        IRenderTarget* VGetTarget(void);

        bool VOnRestore(uint w, uint h, ErrorLog* log = NULL);

        ~Effect(void);
    };

    class EffectChain : public IEffectChain
    {
    private:
        std::vector<std::unique_ptr<IEffect>> m_effects;
        IRenderTarget* m_pSrc;
        IRenderTarget* m_pTarget;
        uint m_w, m_h;
        IEffect* m_leaf;
        IEffectFactory* m_pEffectFactory;
        IShader* m_pVertexShader;

    public:
        EffectChain(IEffectFactory* factroy);

        IEffect* VAppendEffect(const CMShaderDescription& desc, float percentofw = 1.0f, float percentofh = 1.0f);

        void VProcess(void);

        void VSetSource(IRenderTarget* src);

        void VSetTarget(IRenderTarget* target);

        IRenderTarget* VGetResult(void);

        void VOnRestore(uint w, uint h);

        ~EffectChain(void);
    };

    class EffectFactroy : public IEffectFactory
    {
    public:
        IEffect* VCreateEffect(void)
        {
            return new Effect();
        }

        IEffectChain* VCreateEffectChain(void)
        {
            return new EffectChain(CmGetApp()->VGetHumanView()->VGetEffectFactory());
        }
    };
}