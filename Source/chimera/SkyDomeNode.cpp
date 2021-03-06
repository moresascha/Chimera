#include "SceneNode.h"
#include "Components.h"

namespace chimera
{
    IGeometry* CreateSkyDome(void)
    {
        uint segmentsX = 64;
        uint segmentsY = 8;
        uint indexCount = segmentsY * 2 * (segmentsX + 1) + segmentsY;
        uint vertexCount = (segmentsX + 1) * (segmentsY + 1);

        uint* indexBuffer = new uint[indexCount];
        float* vertexBuffer = new float[vertexCount * 5];

        float dphi = 2 * XM_PI / (float)segmentsX;
        float dtheta = XM_PI / (float)segmentsY;
        float phi = 0;
        float theta = 0;
        uint ic = 0;
        uint vc = 0;

        for(uint i = 0; i <= segmentsY; ++i)
        {
            for(uint j = 0; j <= segmentsX; ++j)
            {
                vertexBuffer[vc++] = sin(theta) * cos(phi);
                vertexBuffer[vc++] = cos(1.005 * theta);
                vertexBuffer[vc++] = sin(theta) * sin(phi);

                vertexBuffer[vc++] = phi / XM_2PI;
                vertexBuffer[vc++] = 2 * (XM_PI + theta) / XM_PI;
                phi += dphi;
            }
            phi = 0;
            theta -= 0.5f * dtheta;
        }

        for(uint i = 0; i < segmentsY; ++i)
        {
            for(uint j = 0; j <= segmentsX; ++j)
            {
                indexBuffer[ic++] = i * (segmentsX+1) + j + segmentsX + 1;
                indexBuffer[ic++] = i * (segmentsX+1) + j;
            }
            indexBuffer[ic++] = -1;
        }

        IGeometry* m_sSkyDome = CmGetApp()->VGetHumanView()->VGetGraphicsFactory()->VCreateGeoemtry().release();
        m_sSkyDome->VSetIndexBuffer(indexBuffer, indexCount);
        m_sSkyDome->VSetTopology(eTopo_TriangleStrip);
        m_sSkyDome->VSetVertexBuffer(vertexBuffer, vertexCount, 5 * sizeof(float));
        m_sSkyDome->VCreate();

        delete[] indexBuffer;
        delete[] vertexBuffer;

        return m_sSkyDome;
    }

    SkyDomeNode::SkyDomeNode(ActorId id, chimera::CMResource res) : SceneNode(id), m_TextureRes(res)
    {
        VSetRenderPaths(CM_RENDERPATH_SKY);
    }

    bool SkyDomeNode::VIsVisible(ISceneGraph* graph)
    {
        return true;
    }

    void SkyDomeNode::VOnRestore(ISceneGraph* graph)
    {
        GetActorCompnent<TransformComponent>(m_actor, CM_CMP_TRANSFORM)->GetTransformation()->SetScale(1000);
        m_textureHandle = std::static_pointer_cast<IDeviceTexture>(CmGetApp()->VGetHumanView()->VGetVRamManager()->VGetHandle(m_TextureRes));

        if(!m_pGeometry)
        {
            m_pGeometry = std::shared_ptr<IGeometry>(CreateSkyDome());
        }
    }

    void SkyDomeNode::_VRender(ISceneGraph* graph, chimera::RenderPath& path)
    {
        switch(path)
        {
        case CM_RENDERPATH_SKY :
            {
                if(m_textureHandle->VIsReady())
                {
                    m_textureHandle->VUpdate();
                    //LOG_CRITICAL_ERROR("remember me, this code is untested!, @sky.hlsl");
                    //tbd::TransformComponent* tc = m_actor->GetComponent<tbd::TransformComponent>(tbd::TransformComponent::COMPONENT_ID).lock().get();
                    //util::ICamera* cam = graph->GetCamera().get();
                    //tc->GetTransformation()->SetTranslate(cam->GetEyePos().x, cam->GetEyePos().y, cam->GetEyePos().z);
                    IRenderer* renderer = CmGetApp()->VGetHumanView()->VGetRenderer();
                    renderer->VPushWorldTransform(*VGetTransformation());
                    renderer->VSetTexture(eDiffuseColorSampler, m_textureHandle.get());
                    //chimera::GetContext()->OMSetDepthStencilState(chimera::m_pDepthCmpStencilState, 0);
                    m_pGeometry->VBind();
                    m_pGeometry->VDraw();
                    
                }
                else
                {
                    m_textureHandle = std::static_pointer_cast<IDeviceTexture>(CmGetApp()->VGetHumanView()->VGetVRamManager()->VGetHandle(m_TextureRes));
                }
            } break;
        }

    }

    SkyDomeNode::~SkyDomeNode(void)
    {
        m_pGeometry->VDestroy();
    }
}


