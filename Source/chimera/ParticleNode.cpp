#include "ParticleNode.h"
#include "Resources.h"
#include "GameApp.h"
#include "Cudah.h"
#include "Particles.cuh"
#include "GeometryFactory.h"
#include "SceneGraph.h"
#include "D3DRenderer.h"
#include "SceneGraph.h"
#include "GameView.h"
#include "VRamManager.h"
#include "Camera.h"
#include "Frustum.h"
#include "Components.h"
#include "EventManager.h"
#include "FileSystem.h"

namespace chimera
{
    ParticleNode::ParticleNode(ActorId id) : SceneNode(id), m_pParticleSystem(NULL), m_time(0)
    {
    }

    void ParticleNode::_VRender(chimera::SceneGraph* graph, chimera::RenderPath& path)
    {
        if(m_pParticleSystem->VIsReady())
        {
            m_pParticleSystem->Update();
            std::shared_ptr<chimera::Geometry> geo = m_pParticleSystem->GetGeometry();

            switch(path)
            {
            case eRenderPath_DrawEditMode : 
                {

                    DrawAnchorSphere(m_actor, GetTransformation(), 0.25f);
                    /*
                    app::g_pApp->GetHumanView()->GetRenderer()->SetNormalMapping(FALSE);
                    util::Mat4 m(*m_transformation->GetTransformation());
                    util::Vec3 scale = m_pParticleSystem->GetAxisAlignedBB().GetMax() - m_pParticleSystem->GetAxisAlignedBB().GetMin();
                    scale.Scale(0.5f);
                    m.SetScale(scale.x, scale.y, scale.z);
                    m.Translate(m_pParticleSystem->GetAxisAlignedBB().GetMiddle().x, m_pParticleSystem->GetAxisAlignedBB().GetMiddle().y, m_pParticleSystem->GetAxisAlignedBB().GetMiddle().z);
                    app::g_pApp->GetHumanView()->GetRenderer()->VPushWorldTransform(m);
                    app::g_pApp->GetHumanView()->GetRenderer()->SetDefaultMaterial();
                    app::g_pApp->GetHumanView()->GetRenderer()->SetActorId(m_actor->GetId());
                    GeometryFactory::GetGlobalDefShadingCube()->Bind();
                    GeometryFactory::GetGlobalDefShadingCube()->Draw(); */
                    break;
                }
            case eRenderPath_DrawPicking :
                {
                    DrawPickingSphere(m_actor, GetTransformation(), 1);
                    //DrawPickingCube(m_actor, m_transformation->GetTransformation(), m_pParticleSystem->GetAxisAlignedBB());
                } break;
            case eRenderPath_DrawBounding : 
                {
                    DrawBox(m_pParticleSystem->GetAxisAlignedBB());
                } break;
                /*
            case eDRAW_TO_SHADOW_MAP :
                {
                    d3d::ShaderProgram::GetProgram("PointLightShadowMap_Particles")->Bind();

                    util::Mat4 model;
                    model.RotateX(graph->GetCamera()->GetTheta());
                    model.RotateY(graph->GetCamera()->GetPhi());
                    //model = util::Mat4::Mul(*m_transformation->GetTransformation(), model);
                    app::g_pApp->GetHumanView()->GetRenderer()->VPushWorldTransform(model);
                    geo->SetInstanceBuffer(m_pParticleSystem->GetParticleBuffer());
                    geo->Bind();
                    geo->Draw();
                    geo->Update();

                    d3d::ShaderProgram::GetProgram("PointLightShadowMap")->Bind();

                    break;
                } */
            case eRenderPath_DrawDebugInfo:
                {
                    DrawActorInfos(m_actor, GetTransformation(), graph->GetCamera());
                } break;
            case eRenderPath_DrawParticles : 
            {

                util::Mat4 model;
                model.RotateX(graph->GetCamera()->GetTheta());
                model.RotateY(graph->GetCamera()->GetPhi());
                //model = util::Mat4::Mul(*m_transformation->GetTransformation(), model);
                chimera::g_pApp->GetHumanView()->GetRenderer()->VPushWorldTransform(model);
                geo->SetInstanceBuffer(m_pParticleSystem->GetParticleBuffer());
                geo->Bind();
                geo->Draw();
                geo->Update();
                break;
            }
            }
        }
        else
        {
            VOnRestore(graph);
        }
    }

    bool ParticleNode::VIsVisible(SceneGraph* graph)
    {
        bool in = graph->GetFrustum()->IsInside(m_transformedBBPoint, m_pParticleSystem->GetAxisAlignedBB().GetRadius());
        return in;
    }

    void ParticleNode::VOnUpdate(ulong millis, SceneGraph* graph)
    {
        VOnActorMoved();
        if(m_pParticleSystem->VIsReady())
        {
            if(VIsVisible(graph))
            {
                m_timer.Tick();
                m_pParticleSystem->UpdateTick(m_timer.GetTime(), millis);
                SceneNode::VOnUpdate(millis, graph);
            }
        }
        else
        {
            VOnRestore(graph);
        }
    }

    void Hover(ulong dt, std::shared_ptr<chimera::Actor> a)
    {
        static float time = 0;
        time += dt * 1e-3f;
        chimera::TransformComponent* tc = a->GetComponent<chimera::TransformComponent>(chimera::TransformComponent::COMPONENT_ID).lock().get();
        util::Vec3 t;
        t.y = 0.05f * sin(time);
        t.x = 0.05f * cos(time);
        t.z = 0.05f * sin(time) * cos(time);
        tc->GetTransformation()->Translate(t);
        QUEUE_EVENT(new chimera::ActorMovedEvent(a));
    }

    chimera::GradientField* gf;
    void ParticleNode::OnFileChanged(void)
    {
        gf->VOnRestore(m_pParticleSystem.get());
    }

    void ParticleNode::VOnRestore(chimera::SceneGraph* graph)
    {
        if(m_pParticleSystem == NULL || !m_pParticleSystem->VIsReady())
        {
            util::AxisAlignedBB aabb;
            float bounds = 50;
            aabb.AddPoint(util::Vec3(-bounds, -bounds, -bounds));
            aabb.AddPoint(util::Vec3(+bounds, bounds, +bounds));
            aabb.Construct();

            chimera::BaseEmitter* emitter;// = new SurfaceEmitter("torus.obj", util::Vec3(0,0.1f,0), (UINT)(1.5f * (FLOAT)(1 << 19)), 0, 15000);

            emitter = new chimera::BoxEmitter(util::Vec3(0.5f, 0.1f, 0.5f), util::Vec3(0,0.1f,0), 1024 * 1024, 0, 10); //6000 ((FLOAT)(1 << 8)

            m_pParticleSystem = std::shared_ptr<chimera::ParticleSystem>(new chimera::ParticleSystem(emitter));

            /*tbd::BaseModifier* mod = new tbd::Gravity(-9.81f, 1);

            m_pParticleSystem->AddModifier(mod);

            //mod = new tbd::Turbulence(2, 9.81f);

            //m_pParticleSystem->AddModifier(mod);

            m_pParticleSystem->AddModifier(new tbd::GravityField(util::Vec3(10,10,-10), 1, 5, eAttract));

            m_pParticleSystem->AddModifier(new tbd::GravityField(util::Vec3(10,10,10), 1, 5, eRepel));

            m_pParticleSystem->AddModifier(new tbd::VelocityDamper(0.995f)); */
            gf = new GradientField();
            m_pParticleSystem->AddModifier(gf);

            /*
            util::Plane p;
            p.Init(util::Vec3(0,1,0), 0);
            m_pParticleSystem->AddModifier(new tbd::Plane(p)); */
            m_pParticleSystem->AddModifier(new chimera::BoundingBox());

            m_pParticleSystem->SetAxisAlignedBB(aabb);

            m_pParticleSystem->SetTranslation(GetTransformation()->GetTranslation());

            //unique name
            std::stringstream ss;
            ss << "particlesystem";
            ss << m_actorId;
            chimera::g_pApp->GetHumanView()->GetVRamManager()->AppendAndCreateHandle(ss.str(), m_pParticleSystem);

            std::shared_ptr<proc::Process> proc = std::shared_ptr<proc::Process>(new proc::ActorDelegateProcess(Hover, m_actor));
            //app::g_pApp->GetLogic()->AttachProcess(proc);

            chimera::FileSystem* loader = chimera::g_pApp->GetDLLLoader();
            fastdelegate::FastDelegate0<> l = fastdelegate::MakeDelegate(this, &ParticleNode::OnFileChanged);
            loader->RegisterCallback("ParticleData.dll", "../../ParticleData/ParticleData/x64/Debug/", l);

            //m_timer.Reset();
        }

        VOnActorMoved();
    }

    void ParticleNode::VOnActorMoved(void)
    {
        m_transformedBBPoint = util::Mat4::Transform(*GetTransformation(), m_pParticleSystem->GetAxisAlignedBB().GetMiddle());
        m_pParticleSystem->SetTranslation(GetTransformation()->GetTranslation());
    }

    uint ParticleNode::VGetRenderPaths(void)
    {
        return eRenderPath_DrawParticles | eRenderPath_DrawPicking | eRenderPath_DrawBounding | eRenderPath_DrawEditMode;
    }

    ParticleNode::~ParticleNode(void)
    {
    }
}

