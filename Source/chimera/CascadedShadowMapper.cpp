#include "CascadedShadowMapper.h"
#include "GameApp.h"
#include "D3DRenderer.h"
#include "SceneNode.h"
#include "GameLogic.h"
#include "SceneGraph.h"
#include "Camera.h"
#include "Components.h"
#include "EventManager.h"
namespace d3d
{
//#define SHADOW_START_SIZE 1024

#define CSM_DEBUG

#ifndef _DEBUG
//#undef CSM_DEBUG
#endif    

    CascadedShadowMapper::CascadedShadowMapper(UCHAR cascades) 
        : m_cascades(cascades), m_ppTargets(NULL), /*m_camera(1024, 1024, 0.01, 1000),*/ m_pCascadesSettings(NULL), m_ppBlurChain(NULL)
    {

#ifdef CSM_DEBUG
        for(USHORT i = 0; i < m_cascades; ++i)
        {
            m_cascadeCameraActor[i] = app::g_pApp->GetLogic()->VCreateActor("staticcamera.xml");
            std::stringstream ss;
            ss << "cascadeCam";
            ss << i;
            m_cascadeCameraActor[i]->SetName(ss.str());
        }
#endif
        m_lightActorCamera = app::g_pApp->GetLogic()->VCreateActor("rendercamera.xml");
        util::Vec3 lightPos(1.0f,0.3f,-0.2f);
        lightPos.Normalize();
        m_lightActorCamera->GetComponent<tbd::TransformComponent>(tbd::TransformComponent::COMPONENT_ID).lock()->GetTransformation()->SetTranslate(lightPos.x, lightPos.y, lightPos.z);
        m_lightActorCamera->GetComponent<tbd::CameraComponent>(tbd::CameraComponent::COMPONENT_ID).lock()->GetCamera()->MoveToPosition(
            m_lightActorCamera->GetComponent<tbd::TransformComponent>(tbd::TransformComponent::COMPONENT_ID).lock()->GetTransformation()->GetTranslation());
        m_lightActorCamera->SetName("cascadeLightCamera");

        m_viewActor = app::g_pApp->GetLogic()->VCreateActor("rendercamera.xml");
        m_viewActor->SetName("cascadeViewCamera");
        m_viewActor->GetComponent<tbd::TransformComponent>(tbd::TransformComponent::COMPONENT_ID).lock()->GetTransformation()->SetTranslate(0,5,-5);
        m_viewActor->GetComponent<tbd::CameraComponent>(tbd::CameraComponent::COMPONENT_ID).lock()->GetCamera()->MoveToPosition(
            m_viewActor->GetComponent<tbd::TransformComponent>(tbd::TransformComponent::COMPONENT_ID).lock()->GetTransformation()->GetTranslation());


        ADD_EVENT_LISTENER(this, &CascadedShadowMapper::SetSunPositionDelegate, event::SetSunPositionEvent::TYPE);
    }

    BOOL CascadedShadowMapper::OnRestore(VOID)
    {
        UINT startSize = app::g_pApp->GetConfig()->GetInteger("iCSMSize");

        if(!m_ppTargets)
        {
            m_ppTargets = new d3d::RenderTarget*[m_cascades];
            m_ppBlurredTargets = new d3d::RenderTarget*[m_cascades];
            m_ppBlurChain = new d3d::EffectChain*[m_cascades];

            for(UCHAR i = 0; i < m_cascades; ++i)
            {
                m_ppTargets[i] = new d3d::RenderTarget();
                m_ppBlurredTargets[i] = new d3d::RenderTarget();

                UINT dim = startSize / (1 << i);

                m_ppBlurChain[i] = new d3d::EffectChain(m_ppTargets[i], dim, dim);

                d3d::Effect* e0 = m_ppBlurChain[i]->CreateEffect("VSMBlurH", 1, 1);
                e0->SetSource(m_ppTargets[i]);

                d3d::Effect* e1 = m_ppBlurChain[i]->CreateEffect("VSMBlurV", 1, 1);
                e1->AddRequirement(e0);
                e1->SetTarget(m_ppBlurredTargets[i]);
            }
        }

        SAFE_ARRAY_DELETE(m_pCascadesSettings);

        m_pCascadesSettings = new CascadeSettings[m_cascades];

        FLOAT f = 1000;
        
        FLOAT factor = 3.5;
        FLOAT l = 0;
        for(UCHAR i = 0; i < m_cascades; ++i)
        {
            l += (FLOAT)pow(factor, i);
        }
        FLOAT t = 1.0f / l;
        FLOAT sum = 0;

        FLOAT lastZ = 0;

        for(UCHAR i = 1; i <= m_cascades; ++i)
        {
            UINT dim = startSize / (1 << (i-1));
            m_ppTargets[i - 1]->SetClearColor(1000, 1000, 1000, 1000);
            //m_ppTargets[i - 1]->SetMiscflags(D3D11_RESOURCE_MISC_GENERATE_MIPS);
            m_ppTargets[i - 1]->OnRestore(dim, dim, DXGI_FORMAT_R32G32_FLOAT);

            m_pCascadesSettings[i - 1].start = -10;
            sum += pow(factor, (i-1)) * t;
            m_pCascadesSettings[i - 1].end = 14;

            m_ppBlurChain[i - 1]->OnRestore(dim, dim);
        }

        FLOAT cuttOffset = 0;

        m_pCascadesSettings[0].start = 0;//-cuttOffset;
        m_pCascadesSettings[0].end = 16;

        m_pCascadesSettings[1].start = m_pCascadesSettings[0].end - cuttOffset;
        m_pCascadesSettings[1].end = 55;

        m_pCascadesSettings[2].start = m_pCascadesSettings[1].end - cuttOffset;
        m_pCascadesSettings[2].end = 120;

        /*m_cascadesSettings[3].start = 256;
        m_cascadesSettings[3].end = 1000; */

        m_pProgram = d3d::ShaderProgram::GetProgram("CSM").get();

        m_pProgramInstanced = d3d::ShaderProgram::GetProgram("CSM_Instanced").get();

        util::Vec3 eyePos;
        eyePos = m_lightActorCamera->GetComponent<tbd::TransformComponent>(tbd::TransformComponent::COMPONENT_ID).lock()->GetTransformation()->GetTranslation();
        //m_camera.LookAt(eyePos, util::Vec3(0,0,0));

        std::shared_ptr<tbd::CameraComponent> cc = m_lightActorCamera->GetComponent<tbd::CameraComponent>(tbd::CameraComponent::COMPONENT_ID).lock();
        cc->GetCamera()->LookAt(eyePos, util::Vec3(0,0,0));

        //m_camera.MoveToPosition(eyePos);
        //m_camera.Rotate(0, 0.785f);
        //m_camera.Rotate(0, XM_PIDIV2);
        //m_camera.GetView().Print();
        return TRUE;
    }

    VOID CascadedShadowMapper::Render(tbd::SceneGraph* graph)
    {

        D3DRenderer* renderer = app::g_pApp->GetHumanView()->GetRenderer();
        ID3D11ShaderResourceView* vn[4];
        for(UCHAR i = 0; i < m_cascades + 1; ++i)
        {
            vn[i] = NULL;
        }
        d3d::GetContext()->PSSetShaderResources(d3d::eEffect0, m_cascades, vn + 1); //debugging samplers

        util::ICamera* playerView = graph->GetCamera().get();

        tbd::Frustum cascadeFrusta;
        tbd::Frustum ortographicFrusta;

        std::shared_ptr<tbd::CameraComponent> lcc = m_lightActorCamera->GetComponent<tbd::CameraComponent>(tbd::CameraComponent::COMPONENT_ID).lock();

        std::shared_ptr<tbd::CameraComponent> vcc = m_viewActor->GetComponent<tbd::CameraComponent>(tbd::CameraComponent::COMPONENT_ID).lock();
        
        renderer->VPushViewTransform(lcc->GetCamera()->GetView(), lcc->GetCamera()->GetIView(), lcc->GetCamera()->GetEyePos());

        util::ICamera* lightCamera = lcc->GetCamera().get();
        //util::ICamera* viewCamera = vcc->GetCamera().get();
        util::ICamera* viewCamera = playerView;

        //util::Mat4 viewToLight = util::Mat4::Mul(lightCamera->GetView(), viewCamera->GetIView());//vcc->GetCamera()->GetIView());
        //renderer->PushRasterizerState(d3d::g_pRasterizerStateBackFaceSolid);

        FLOAT distances[3];

        for(UCHAR ci = 0; ci < m_cascades; ++ci)
        {
            CascadeSettings& settings = m_pCascadesSettings[ci];
            
#ifdef CSM_DEBUG
            util::StaticCamera* staticCam;
            staticCam = (util::StaticCamera*)(m_cascadeCameraActor[ci]->GetComponent<tbd::CameraComponent>(tbd::CameraComponent::COMPONENT_ID).lock()->GetCamera().get());
#endif

            FLOAT farr = settings.end;
            FLOAT nnear = settings.start;
 
            cascadeFrusta.CreatePerspective(viewCamera->GetAspect(), viewCamera->GetFoV(), nnear, farr);

            util::Vec3 vmin(FLT_MAX, FLT_MAX, FLT_MAX);
            util::Vec3 vmax(FLT_MIN, FLT_MIN, FLT_MIN);
            util::Vec3 vFrustumPoints[8];
            util::Vec3 vecs[8];

            for(UINT i = 0; i < 8; ++i)
            {
                util::Vec3 v = util::Mat4::Transform(viewCamera->GetIView(), cascadeFrusta.GetPoints()[i]);

                vFrustumPoints[i] = v;

                util::Vec3 pos = util::Mat4::Transform(lightCamera->GetView(), v);

                vmax = util::Vec3::Max(vmax, pos);
                vmin = util::Vec3::Min(vmin, pos);
            }

            /*FLOAT bound = (farr - nnear) / 1024.0f;

            util::Vec3 fmin((INT)vmin.x / bound, (INT)vmin.y / bound, (INT)vmin.y / bound);
            fmin.Scale(bound);
            vmin = fmin;

            util::Vec3 fmax((INT)(vmax.x / bound), (INT)(vmax.y / bound), (INT)(vmax.y / bound));
            fmax.Scale(bound);
            vmax = fmax; */

            /*vmax = util::Mat4::Transform(lightCamera->GetIView(), vmax);
            vmin = util::Mat4::Transform(lightCamera->GetIView(), vmin); */
            /*util::Vec3 vDiagonal = vFrustumPoints[tbd::rightUpNear] - vFrustumPoints[tbd::leftDownFar];
            FLOAT l = vDiagonal.Length();
            vDiagonal.Set(l, l, l);
            // The offset calculated will pad the ortho projection so that it is always the same size 
            // and big enough to cover the entire cascade interval.
            util::Vec3 diff = vmax - vmin;
            util::Vec3 vBoarderOffset = vDiagonal - diff;
            vBoarderOffset.Scale(0.5f);

            vBoarderOffset.z = 0;
            // Add the offsets to the projection.
            vmax = vmax + vBoarderOffset;
            vmin = vmin - vBoarderOffset; */
            
            float n = min(-120, vmin.z); //todo
            float ff = vmax.z;

            distances[ci] = ff - n;

            XMMATRIX mat = XMMatrixOrthographicOffCenterLH(vmin.x, vmax.x, vmin.y, vmax.y, n, ff);

            XMStoreFloat4x4(&settings.m_projection.m_m, mat);

#ifdef CSM_DEBUG
            staticCam->SetOrthographicProjectionOffCenter(vmin.x, vmax.x, vmin.y, vmax.y, n, ff);
            staticCam->SetView(lightCamera->GetView(), lightCamera->GetIView());
#endif
            //ortographicFrusta.CreateOrthographicOffCenter(vmin.x, vmax.x, vmin.y, vmax.y, -120, ff);
            ortographicFrusta = staticCam->GetFrustum();
   
            renderer->VPushProjectionTransform(settings.m_projection, ff - n);

            m_ppTargets[ci]->Clear();
            m_ppTargets[ci]->Bind();

            graph->PushFrustum(&ortographicFrusta);
            m_pProgram->Bind();
            graph->OnRender(tbd::eDRAW_TO_SHADOW_MAP);
            m_pProgramInstanced->Bind();
            graph->OnRender(tbd::eDRAW_TO_SHADOW_MAP_INSTANCED);
            graph->PopFrustum();

            renderer->VPopProjectionTransform();

            m_ppBlurChain[ci]->Process();

         //   d3d::GetContext()->GenerateMips(m_ppBlurredTargets[ci]->GetShaderRessourceView());
        }

       // renderer->PopRasterizerState();
        renderer->VPopViewTransform();
        d3d::BindBackbuffer();
        ID3D11ShaderResourceView* v[3];
        for(UCHAR i = 0; i < m_cascades; ++i)
        {
            v[i] = m_ppBlurredTargets[i]->GetShaderRessourceView();
        }
        d3d::GetContext()->PSSetShaderResources(d3d::eEffect0, m_cascades, v);
        //ID3D11ShaderResourceView* debugView = m_ppBlurredTargets[0]->GetShaderRessourceView();
        //d3d::GetContext()->PSSetShaderResources(d3d::eEffect3, 1, &debugView); //debugging samplers
        
        util::Mat4 mats[3]; //TODO
        //FLOAT distances[3];
        for(UCHAR i = 0; i < m_cascades; ++i)
        {
            mats[i] = m_pCascadesSettings[i].m_projection;
        }
        
        renderer->SetCSMSettings(lightCamera->GetView(), lightCamera->GetIView(), mats, lcc->GetCamera()->GetEyePos(), distances);
    }

    VOID CascadedShadowMapper::SetSunPositionDelegate(event::IEventPtr data)
    {
        std::shared_ptr<event::SetSunPositionEvent> e = std::static_pointer_cast<event::SetSunPositionEvent>(data);
        std::shared_ptr<tbd::CameraComponent> cc = m_lightActorCamera->GetComponent<tbd::CameraComponent>(tbd::CameraComponent::COMPONENT_ID).lock();
        util::Vec3 newPos = e->m_position;
        newPos.Normalize();
        cc->GetCamera()->LookAt(newPos, util::Vec3(0,0,0));
        util::Vec3 f(0,0,0);
        util::Vec3 u(0,1,0);
        util::Mat4 m;
        XMStoreFloat4x4(&m.m_m, XMMatrixLookAtLH(XMLoadFloat3(&newPos.m_v), XMLoadFloat3(&(f.m_v)), XMLoadFloat3(&u.m_v)));
    }

    VOID CascadedShadowMapper::Destroy(VOID)
    {
        for(UCHAR i = 0; i < m_cascades; ++i)
        {
            SAFE_DELETE(m_ppTargets[i]);
            SAFE_DELETE(m_ppBlurredTargets[i]);
            SAFE_DELETE(m_ppBlurChain[i]);
        }

        SAFE_ARRAY_DELETE(m_ppTargets);

        SAFE_ARRAY_DELETE(m_ppBlurredTargets);

        SAFE_ARRAY_DELETE(m_ppBlurChain);

        SAFE_ARRAY_DELETE(m_pCascadesSettings);
    }

    CascadedShadowMapper::~CascadedShadowMapper(VOID)
    {
        REMOVE_EVENT_LISTENER(this, &CascadedShadowMapper::SetSunPositionDelegate, event::SetSunPositionEvent::TYPE);
        Destroy();
    }
};
