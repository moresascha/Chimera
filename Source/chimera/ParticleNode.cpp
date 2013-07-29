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
namespace tbd
{
    ParticleNode::ParticleNode(ActorId id) : SceneNode(id), m_pParticleSystem(NULL), m_time(0)
    {
    }

    VOID ParticleNode::_VRender(tbd::SceneGraph* graph, tbd::RenderPath& path)
    {
        if(m_pParticleSystem->IsReady())
        {
            m_pParticleSystem->Update();
            std::shared_ptr<d3d::Geometry> geo = m_pParticleSystem->GetGeometry();

            switch(path)
            {
            case eDRAW_EDIT_MODE : 
                {

                    DrawAnchorSphere(m_actor, m_transformation->GetTransformation(), 0.25f);
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
            case eDRAW_PICKING :
                {
                    DrawPickingSphere(m_actor, m_transformation->GetTransformation(), 1);
                    //DrawPickingCube(m_actor, m_transformation->GetTransformation(), m_pParticleSystem->GetAxisAlignedBB());
                } break;
            case eDRAW_BOUNDING_DEBUG : 
                {
                    DrawBox(m_transformation->GetTransformation(), m_pParticleSystem->GetAxisAlignedBB());
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
            case eDRAW_DEBUG_INFOS:
                {
                    DrawActorInfos(m_actor, m_transformation->GetTransformation(), graph->GetCamera());
                } break;
            case eDRAW_PARTICLE_EFFECTS : 
            {

                util::Mat4 model;
                model.RotateX(graph->GetCamera()->GetTheta());
                model.RotateY(graph->GetCamera()->GetPhi());
                //model = util::Mat4::Mul(*m_transformation->GetTransformation(), model);
                app::g_pApp->GetHumanView()->GetRenderer()->VPushWorldTransform(model);
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

    BOOL ParticleNode::VIsVisible(SceneGraph* graph)
    {
        BOOL in = graph->GetFrustum()->IsInside(m_transformedBBPoint, m_pParticleSystem->GetAxisAlignedBB().GetRadius());
        return in;
    }

    VOID ParticleNode::VOnUpdate(ULONG millis, SceneGraph* graph)
    {
        if(m_pParticleSystem->IsReady())
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

    VOID ParticleNode::VOnRestore(tbd::SceneGraph* graph)
    {
        util::AxisAlignedBB aabb;
        FLOAT bounds = 50;
        aabb.AddPoint(util::Vec3(-bounds, -bounds, -bounds));
        aabb.AddPoint(util::Vec3(+bounds, bounds, +bounds));
        aabb.Construct();
        
        tbd::BaseEmitter* emitter;// = new SurfaceEmitter("torus.obj", util::Vec3(0,0.1f,0), (UINT)(1.5f * (FLOAT)(1 << 19)), 0, 15000);
            
        emitter = new tbd::BoxEmitter(util::Vec3(0.5f, 0.1f, 0.5f), util::Vec3(0,0.1f,0), 192 * 200, 0, 15); //6000 ((FLOAT)(1 << 8)

        m_pParticleSystem = std::shared_ptr<tbd::ParticleSystem>(new tbd::ParticleSystem(emitter));

        tbd::BaseModifier* mod = new tbd::Gravity(-9.81f, 1);

        m_pParticleSystem->AddModifier(mod);

        //mod = new tbd::Turbulence(2, 9.81f);

        //m_pParticleSystem->AddModifier(mod);

        m_pParticleSystem->AddModifier(new tbd::GravityField(util::Vec3(10,10,-10), 1, 1, eAttract));

        m_pParticleSystem->AddModifier(new tbd::GravityField(util::Vec3(10,10,10), 1, 5, eRepel));

        m_pParticleSystem->AddModifier(new tbd::VelocityDamper(0.995f));

        m_pParticleSystem->AddModifier(new GradientField());

        m_pParticleSystem->SetAxisAlignedBB(aabb);

        m_pParticleSystem->SetTranslation(this->m_transformation->GetTransformation()->GetTranslation());

        //unique name
        std::stringstream ss;
        ss << "particlesystem";
        ss << m_actorId;
        app::g_pApp->GetHumanView()->GetVRamManager()->AppendAndCreateHandle(ss.str(), m_pParticleSystem);

        m_timer.Reset();
    }

    VOID ParticleNode::VOnActorMoved(VOID)
    {
        m_transformedBBPoint = util::Mat4::Transform(*this->m_transformation->GetTransformation(), m_pParticleSystem->GetAxisAlignedBB().GetMiddle());
        m_pParticleSystem->SetTranslation(this->m_transformation->GetTransformation()->GetTranslation());
    }

    UINT ParticleNode::VGetRenderPaths(VOID)
    {
        return eDRAW_PARTICLE_EFFECTS | eDRAW_PICKING | eDRAW_BOUNDING_DEBUG | eDRAW_EDIT_MODE;
    }

    ParticleNode::~ParticleNode(VOID)
    {
    }
}

