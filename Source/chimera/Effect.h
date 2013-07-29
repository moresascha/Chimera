#pragma once
#include "stdafx.h"
#include "d3d.h"

namespace d3d
{
    class RenderTarget;
    class PixelShader;
    class VertexShader;

    VOID DrawScreenQuad(INT x, INT y, INT w, INT h);

    VOID DrawLine(INT x, INT y, INT w, INT h);

    class IEffectParmaters
    {
    public:
        virtual VOID VApply(VOID){ LOG_CRITICAL_ERROR("VApply not implemented!"); }
    };

    class DefaultParams : public IEffectParmaters
    {
    public:
        VOID VApply(VOID) {  }
    };

    typedef VOID (*EffectDrawMethod) (VOID);

    class Effect
    {
        friend class EffectChain;
    private:
        BOOL m_ownsTarget;
        d3d::RenderTarget* m_source;
        d3d::RenderTarget* m_target;
        BOOL m_created, m_isProcessed;
        d3d::PixelShader* m_pPixelShader;
        LPCSTR m_pixelShaderFunction;
        FLOAT m_w, m_h;
        std::shared_ptr<IEffectParmaters> m_params;
        std::list<Effect*> m_requirements;
        VOID Process(VOID);
        EffectDrawMethod m_pfDraw;

    public:
        Effect(LPCSTR pixelShader, FLOAT w, FLOAT h);
        VOID SetParameters(std::shared_ptr<IEffectParmaters> params);
        VOID SetDrawMethod(EffectDrawMethod dm);
        VOID AddRequirement(Effect* e);
        VOID SetSource(d3d::RenderTarget* src);
        VOID SetTarget(d3d::RenderTarget* target);
        d3d::RenderTarget* GetTarget(VOID);
        BOOL OnRestore(UINT w, UINT h, ErrorLog* log = NULL);
        ~Effect(VOID);
    };

    class EffectChain
    {
    private:
        std::list<d3d::Effect*> m_effects;
        d3d::RenderTarget* m_src;
        UINT m_w, m_h;
        Effect* m_leaf;
    public:

        EffectChain(d3d::RenderTarget* src, UINT w, UINT h);

        Effect* CreateEffect(LPCSTR pixelShader, FLOAT percentofw = 1.0f, FLOAT percentofh = 1.0f);

        VOID Process(VOID);

        VOID OnRestore(UINT w, UINT h);

        ~EffectChain(VOID);
        
        static d3d::VertexShader* m_spScreenQuadVShader;
        static d3d::PixelShader* m_spScreenQuadPShader;

        static BOOL StaticCreate(VOID);
        static VOID StaticDestroy(VOID);
    };
}