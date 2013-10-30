#pragma once
#include "stdafx.h"

namespace chimera
{
    class DefaultParams : public IEffectParmaters
    {
    public:
        VOID VApply(VOID) {  }
    };

    class Effect : public IEffect
    {
    private:
        IRenderTarget* m_source;
        IRenderTarget* m_target;
        std::unique_ptr<IRenderTarget> m_ownedTarget;
        BOOL m_created, m_isProcessed;
        IShader* m_pPixelShader;
        FLOAT m_w, m_h;
        IEffectParmaters* m_params; //TODO
        std::vector<IEffect*> m_requirements;
        CMShaderDescription m_shaderDesc;
        EffectDrawMethod m_pfDraw;

    public:
        Effect(VOID);

        VOID VCreate(CONST CMShaderDescription& shaderDesc, FLOAT w, FLOAT h);

        VOID VSetParameters(IEffectParmaters* params);

        VOID VSetDrawMethod(EffectDrawMethod dm);

        VOID VAddRequirement(IEffect* e);

        VOID VReset(VOID);

        VOID VSetSource(IRenderTarget* src);

        FLOAT2 VGetViewPort(VOID);

        VOID VProcess(VOID);

        VOID VSetTarget(IRenderTarget* target);

        VOID VSetTarget(std::unique_ptr<IRenderTarget> target);

        IRenderTarget* VGetTarget(VOID);

        BOOL VOnRestore(UINT w, UINT h, ErrorLog* log = NULL);

        ~Effect(VOID);
    };

    class EffectChain : public IEffectChain
    {
    private:
        std::vector<std::unique_ptr<IEffect>> m_effects;
        IRenderTarget* m_pSrc;
        IRenderTarget* m_pTarget;
        UINT m_w, m_h;
        IEffect* m_leaf;
        IEffectFactory* m_pEffectFactory;
        IShader* m_pVertexShader;

    public:

        EffectChain(IEffectFactory* factroy);

        IEffect* VCreateEffect(CONST CMShaderDescription& desc, FLOAT percentofw = 1.0f, FLOAT percentofh = 1.0f);

        VOID VProcess(VOID);

        VOID VSetSource(IRenderTarget* src);

        VOID VSetTarget(IRenderTarget* target);

        IRenderTarget* VGetResult(VOID);

        VOID VOnRestore(UINT w, UINT h);

        ~EffectChain(VOID);
    };

    class EffectFactroy : public IEffectFactory
    {
    public:
        IEffect* VCreateEffect(VOID)
        {
            return new Effect();
        }

        IEffectChain* VCreateEffectChain(VOID)
        {
            return new EffectChain(CmGetApp()->VGetHumanView()->VGetEffectFactory());
        }
    };
}