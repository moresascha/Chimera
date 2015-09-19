#include "ParticleNode.h"
#include "Event.h"
#include "Frustum.h"
#include "Components.h"
#include "ParticleSystem.h"
#include "util.h"

#define QUAD_GEO_SUFFIX "quadgeo"

namespace chimera
{

    class ParticleSystemHandle : public VRamHandle
    {
    private:
        IParticleSystem* m_pSystem;

    public:
        ParticleSystemHandle(IParticleSystem* system) : m_pSystem(system)
        {

        }

        bool VCreate(void)
        {
            if(m_pSystem)
            {
                m_pSystem->VOnRestore();
            }
            return m_pSystem != NULL;
        }

        void VDestroy()
        {
            if(m_pSystem)
            {
                m_pSystem->VRelease();
            }
        }

        uint VGetByteCount(void) const
        {
            return m_pSystem ? m_pSystem->VGetByteCount() : 0;
        }

        ~ParticleSystemHandle(void)
        {
            SAFE_DELETE(m_pSystem);
        }
    };

    class DefaultParticleGeometry : public IVRamHandleCreator
    {
    public:
        IVRamHandle* VGetHandle(void)
        {
            return CmGetApp()->VGetHumanView()->VGetGraphicsFactory()->VCreateGeoemtry().release();
        }

        void VCreateHandle(IVRamHandle* handle)
        {
            IGeometry* geo = (IGeometry*)handle;

            float qscale = 0.2f;

            float quad[32] = 
            {
                -1 * qscale, -1 * qscale, 0, 0,1,0, 0, 0, 
                +1 * qscale, -1 * qscale, 0, 0,1,0, 1, 0,
                -1 * qscale, +1 * qscale, 0, 0,1,0, 0, 1,
                +1 * qscale, +1 * qscale, 0, 0,1,0, 1, 1
            };

            uint indices[4] = 
            {
                0, 1, 2, 3
            };

            geo->VSetVertexBuffer(quad, 4, 32);
            geo->VSetTopology(eTopo_TriangleStrip);
            geo->VSetIndexBuffer(indices, 4);
            geo->VCreate();
        }
    };

    ParticleNode::ParticleNode(ActorId id, IParticleSystem* pSystem) : SceneNode(id), m_pParticleSystem(pSystem), m_time(0), m_particleSystemHandle(NULL)
    {
        CmGetApp()->VGetHumanView()->VGetVRamManager()->VRegisterHandleCreator(QUAD_GEO_SUFFIX, new DefaultParticleGeometry());
    }

    void ParticleNode::_VRender(ISceneGraph* graph, RenderPath& path)
    {
        if(m_particleSystemHandle->VIsReady())
        {
            if(!m_pGeometry || !m_pGeometry->VIsReady())
            {
                m_pGeometry = std::static_pointer_cast<IGeometry>(CmGetApp()->VGetHumanView()->VGetVRamManager()->VGetHandle(std::string("basicEmitter.") + std::string(QUAD_GEO_SUFFIX)));
            }

            switch(path)
            {
            case CM_RENDERPATH_PARTICLE : 
                {
                    util::Mat4 model;
                    model.RotateX(graph->VGetCamera()->GetTheta());
                    model.RotateY(graph->VGetCamera()->GetPhi());

                    CmGetApp()->VGetRenderer()->VPushWorldTransform(model);

                    for(auto& it = m_pParticleSystem->VGetEmitter().begin(); it != m_pParticleSystem->VGetEmitter().end(); ++it)
                    {
                        m_pGeometry->VSetInstanceBuffer((*it)->VGetGFXPosArray(), 0);
                        m_pGeometry->VSetInstanceBuffer((*it)->VGetGFXVeloArray(), 1);
                        m_pGeometry->VBind();
                        m_pGeometry->VDraw();
                        m_pGeometry->VUpdate();
                    }
                    m_particleSystemHandle->VUpdate();
                    break;
                } break;
            }
            /*
            switch(path)
            {
            case eRenderPath_DrawEditMode : 
                {

                    DrawAnchorSphere(m_actor, VGetTransformation(), 0.25f);
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
                    GeometryFactory::GetGlobalDefShadingCube()->Draw();
                    break;
                }
            case eRenderPath_DrawPicking :
                {
                    DrawPickingSphere(m_actor, VGetTransformation(), 1);
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
                }
            case eRenderPath_DrawDebugInfo:
                {
                    DrawActorInfos(m_actor, VGetTransformation(), graph->VGetCamera());
                } break;
            case eRenderPath_DrawParticles : 
            {

                util::Mat4 model;
                model.RotateX(graph->VGetCamera()->GetTheta());
                model.RotateY(graph->VGetCamera()->GetPhi());
                //model = util::Mat4::Mul(*m_transformation->GetTransformation(), model);
                chimera::g_pApp->GetHumanView()->GetRenderer()->VPushWorldTransform(model);
                geo->SetInstanceBuffer(m_pParticleSystem->VGetParticleBuffer());
                geo->Bind();
                geo->Draw();
                geo->Update();
                break;
            }
            } */
        }
        else
        {
            VOnRestore(graph);
        }
    }

    bool ParticleNode::VIsVisible(ISceneGraph* graph)
    {
        bool in = graph->VGetFrustum()->IsInside(m_transformedBBPoint, m_pParticleSystem->VGetAxisAlignedBB().GetRadius());
        return true;
    }

    void ParticleNode::VOnUpdate(ulong millis, ISceneGraph* graph)
    {
        VOnActorMoved();
        if(m_particleSystemHandle->VIsReady())
        {
            if(VIsVisible(graph))
            {
                m_timer.VTick();
                m_pParticleSystem->VUpdate(m_timer.VGetTime(), millis);
                SceneNode::VOnUpdate(millis, graph);
            }
            m_particleSystemHandle->VUpdate();
        }
        else
        {
            VOnRestore(graph);
        }
    }

    void Hover(ulong dt, IActor* a)
    {
        static float time = 0;
        time += dt * 1e-3f;
        TransformComponent* tc;
        a->VQueryComponent(CM_CMP_TRANSFORM, (IActorComponent**)&tc);
        util::Vec3 t;
        t.y = 0.05f * sin(time);
        t.x = 0.05f * cos(time);
        t.z = 0.05f * sin(time) * cos(time);
        tc->GetTransformation()->Translate(t);
        QUEUE_EVENT(new ActorMovedEvent(a));
    }

    void ParticleNode::VOnRestore(ISceneGraph* graph)
    {
        if(!m_particleSystemHandle || !m_particleSystemHandle->VIsReady())
        {
            std::stringstream ss;
            ss << "ParticleSystem_";
            ss << m_actorId;

            m_pParticleSystem = new ParticleSystem(1000000);

            util::cmRNG rng;
            for(int i = 0; i < 1; ++i)
            {
                std::unique_ptr<IParticleEmitter> emit(new BaseEmitter(1e5, 0, 10));
                emit->VSetPosition(util::Vec3(rng.NextFloat(10), 10, rng.NextFloat(10)));
                m_pParticleSystem->VAddEmitter(emit);
            }
            std::unique_ptr<IParticleModifier> gravity(new Gravity(util::Vec3(-0.001f, 0.001f, -0.001)));
            //m_pParticleSystem->VAddModifier(gravity);

            std::unique_ptr<IParticleModifier> turbo(new Turbulence(1, 0.01f));
            //m_pParticleSystem->VAddModifier(turbo);

            std::unique_ptr<IParticleModifier> gfield(new GradientField());
            m_pParticleSystem->VAddModifier(gfield);

            m_particleSystemHandle = std::shared_ptr<ParticleSystemHandle>(new ParticleSystemHandle(m_pParticleSystem));

            m_particleSystemHandle->VSetResource(ss.str());

            CmGetApp()->VGetHumanView()->VGetVRamManager()->VAppendAndCreateHandle(m_particleSystemHandle);
        }

        VOnActorMoved();
    }

    void ParticleNode::VOnActorMoved(void)
    {
        m_transformedBBPoint = util::Mat4::Transform(*VGetTransformation(), m_pParticleSystem->VGetAxisAlignedBB().GetMiddle());
        //m_pParticleSystem->SetTranslation(GetTransformation()->GetTranslation());
    }

    ParticleNode::~ParticleNode(void)
    {
    }
}

