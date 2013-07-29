#include "Effect.h"
#include "GeometryFactory.h"
#include "D3DRenderer.h"
#include "d3d.h"
#include "RenderTarget.h"
#include "ShaderProgram.h"

#ifdef _DEBUG
#include "Process.h"
#include "GameApp.h"
#include "GameLogic.h"
#endif

namespace d3d
{
    VOID DrawScreenQuad(INT x, INT y, INT w, INT h)
    {
        d3d::EffectChain::m_spScreenQuadVShader->Bind();
        d3d::EffectChain::m_spScreenQuadPShader->Bind();

        FLOAT _x = -1.0f + 2 * x / (FLOAT)d3d::g_width;
        FLOAT _y = -1.0f + 2 * y / (FLOAT)d3d::g_height;
        FLOAT _w = _x + 2 * w / (FLOAT)d3d::g_width;
        FLOAT _h = _y + 2 * h / (FLOAT)d3d::g_height;
        FLOAT localVertices[20] = 
        {
            _x, _y, 0, 0, 1,
            _w, _y, 0, 1, 1,
            _x, _h, 0, 0, 0,
            _w, _h, 0, 1, 0,
        };

        d3d::Geometry* quad = GeometryFactory::GetGlobalScreenQuadCPU();
        D3D11_MAPPED_SUBRESOURCE* ress = quad->GetVertexBuffer()->Map();
        memcpy(ress->pData, localVertices, 20 * sizeof(FLOAT));
        quad->GetVertexBuffer()->Unmap();
        quad->Bind();
        quad->Draw();
    }

    VOID DrawLine(INT x, INT y, INT w, INT h)
    {
        FLOAT _x = -1.0f + 2 * x / (FLOAT)d3d::g_width;
        FLOAT _y = -1.0f + 2 * y / (FLOAT)d3d::g_height;
        FLOAT _w = _x + 2 * w / (FLOAT)d3d::g_width;
        FLOAT _h = _y + 2 * h / (FLOAT)d3d::g_height;
        FLOAT localVertices[10] = 
        {
            _x, _y, 0, 0, 0,
            _w, _h, 0, 1, 1,
        };

        d3d::Geometry* line = GeometryFactory::GetGlobalLineCPU();
        D3D11_MAPPED_SUBRESOURCE* ress = line->GetVertexBuffer()->Map();
        memcpy(ress->pData, localVertices, 10 * sizeof(FLOAT));
        line->GetVertexBuffer()->Unmap();
        line->Bind();
        line->Draw();
    }

    VOID DefaultDraw(VOID)
    {
        GeometryFactory::GetGlobalScreenQuad()->Bind();
        GeometryFactory::GetGlobalScreenQuad()->Draw();
    }

    d3d::VertexShader* EffectChain::m_spScreenQuadVShader = NULL;
    d3d::PixelShader* EffectChain::m_spScreenQuadPShader = NULL;

    Effect::Effect(LPCSTR pixelShader, FLOAT w, FLOAT h) 
        : m_pPixelShader(NULL), m_target(NULL), m_source(NULL), m_pixelShaderFunction(pixelShader), m_w(w), m_h(h), 
        m_params(std::shared_ptr<DefaultParams>(new DefaultParams())), m_isProcessed(FALSE), m_ownsTarget(FALSE)
    {
        m_pfDraw = DefaultDraw;
    }

    VOID Effect::SetDrawMethod(EffectDrawMethod dm)
    {
        m_pfDraw = dm;
    }

    BOOL Effect::OnRestore(UINT w, UINT h, ErrorLog* log)
    {
        if(!m_pPixelShader)
        {
            m_pPixelShader = d3d::PixelShader::CreateShader(m_pixelShaderFunction, L"Effects.hlsl", m_pixelShaderFunction);
        }

        if(m_target)
        {
            m_target->OnRestore((UINT)(w * m_w), (UINT)(h * m_h), DXGI_FORMAT_R32G32B32A32_FLOAT, FALSE);
        }

        return TRUE;
    }

    VOID Effect::AddRequirement(Effect* e)
    {
        e->m_target = new d3d::RenderTarget();
        e->m_ownsTarget = TRUE;
        e->m_target->OnRestore((UINT)(e->m_w * (FLOAT)d3d::g_width), (UINT)(e->m_h * (FLOAT)d3d::g_height), DXGI_FORMAT_R32G32B32A32_FLOAT, FALSE);
        m_requirements.push_back(e);
    }

    VOID Effect::SetParameters(std::shared_ptr<IEffectParmaters> params)
    {
        m_params = params;
    }

    VOID Effect::SetSource(d3d::RenderTarget* src)
    {
        m_source = src;
    }

    d3d::RenderTarget* Effect::GetTarget(VOID)
    {
        return m_target;
    }

    VOID Effect::SetTarget(d3d::RenderTarget* target)
    {
        m_target = target;
    }

    VOID Effect::Process(VOID)
    {
        if(m_isProcessed)
        {
            return;
        }

        for(auto it = m_requirements.begin(); it != m_requirements.end(); ++it)
        {
            (*it)->Process();
        }

        m_pPixelShader->Bind();

        if(m_target == NULL)
        {
            d3d::BindBackbuffer();
        }
        else
        {
            m_target->Clear();
            m_target->Bind();
        }

        m_params->VApply();

        CONST UINT c_startSlot = d3d::eEffect0;
        UINT startSlot = c_startSlot;
        ID3D11ShaderResourceView* view = NULL;

        if(m_source)
        {
            view = m_source->GetShaderRessourceView();
            d3d::GetContext()->PSSetShaderResources(startSlot++, 1, &view); //TODO move slot to renderer?
        }

        for(auto it = m_requirements.begin(); it != m_requirements.end(); ++it)
        {
            Effect* e = (*it);
            e->Process();
            view = e->m_target->GetShaderRessourceView();
            d3d::GetContext()->PSSetShaderResources(startSlot++, 1, &view); //TODO move slot to renderer?
        }

        m_pfDraw();

        view = NULL;

        for(INT i = startSlot-1; i >= c_startSlot; --i)
        {
            d3d::GetContext()->PSSetShaderResources(i, 1, &view); //TODO move slot to renderer?
        }

        m_isProcessed = TRUE;
    }

    Effect::~Effect(VOID)
    {
        if(m_ownsTarget)
        {
            SAFE_DELETE(m_target);
        }
    }

    EffectChain::EffectChain(d3d::RenderTarget* src, UINT w, UINT h) : m_src(src), m_w(w), m_h(h)
    {
    }

    Effect* EffectChain::CreateEffect(LPCSTR pixelShader, FLOAT w, FLOAT h)
    {
        Effect* e = new Effect(pixelShader, w, h);

        ErrorLog log;
        if(!e->OnRestore(d3d::g_width, d3d::g_height, &log))
        {
            LOG_CRITICAL_ERROR(log.c_str()); //we check on creation for errors
        }

        m_leaf = e;

        m_effects.push_back(e);

        return e;
    }

    VOID EffectChain::OnRestore(UINT w, UINT h)
    {
        for(auto it = m_effects.begin(); it != m_effects.end(); ++it)
        {
            Effect* effect = *it;
            effect->OnRestore(w, h);
        }
    }

    EffectChain::~EffectChain(VOID)
    {
        for(auto it = m_effects.begin(); it != m_effects.end(); ++it)
        {
            Effect* e = *it;
            delete e;
        }
    }

    VOID EffectChain::Process(VOID)
    {
        
        //ID3D11VertexShader* tmp;
        //d3d::GetContext()->VSGetShader(&tmp, NULL, 0);

        m_spScreenQuadVShader->Bind();

        m_leaf->Process();

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
            (*it)->m_isProcessed = FALSE;
        }

        //d3d::GetContext()->VSSetShader(tmp, NULL, 0);
    }

    BOOL EffectChain::StaticCreate(VOID)
    {
        m_spScreenQuadVShader = d3d::VertexShader::CreateShader("Effect_VS", L"Effects.hlsl", "Effect_VS");
        m_spScreenQuadVShader->SetInputAttr("POSITION", 0, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        m_spScreenQuadVShader->SetInputAttr("TEXCOORD", 1, 0, DXGI_FORMAT_R32G32_FLOAT);
        m_spScreenQuadVShader->GenerateLayout();
        m_spScreenQuadPShader = d3d::PixelShader::CreateShader("Effect_PS", L"Effects.hlsl", "Sample0");
        return TRUE;
    }

    VOID EffectChain::StaticDestroy(VOID)
    {

    }
}
