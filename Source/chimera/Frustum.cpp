#include "Frustum.h"
#include "Line.h"
#include <limits>

const UCHAR INS = 0 << 1;
const UCHAR LEFT = 1 << 1;
const UCHAR RIGHT = 1 << 2;
const UCHAR UP = 1 << 3;
const UCHAR DOWN = 1 << 4;
const UCHAR FRONT = 1 << 5;
const UCHAR BACK = 1 << 6;

namespace chimera
{

    void Frustum::CreateOrthographicOffCenter(float l, float r, float b, float t, float fNear, float fFar)
    {
        m_Initpoints[rightUpNear] = util::Vec3(r, t, fNear);
        m_Initpoints[rightDownNear] = util::Vec3(r, b, fNear);
        m_Initpoints[leftDownNear] = util::Vec3(l, b, fNear);
        m_Initpoints[leftUpNear] = util::Vec3(l, t, fNear);

        m_Initpoints[rightUpFar] = util::Vec3(r, t, fFar);
        m_Initpoints[rightDownFar] = util::Vec3(r, b, fFar);
        m_Initpoints[leftDownFar] = util::Vec3(l, b, fFar);
        m_Initpoints[leftUpFar] = util::Vec3(l, t, fFar);

        m_points[rightUpNear] = util::Vec3(r, t, fNear);
        m_points[rightDownNear] = util::Vec3(r, b, fNear);
        m_points[leftDownNear] = util::Vec3(l, b, fNear);
        m_points[leftUpNear] = util::Vec3(l, t, fNear);

        m_points[rightUpFar] = util::Vec3(r, t, fFar);
        m_points[rightDownFar] = util::Vec3(r, b, fFar);
        m_points[leftDownFar] = util::Vec3(l, b, fFar);
        m_points[leftUpFar] = util::Vec3(l, t, fFar);

        CreatePlanes();
    }

    void Frustum::CreateOrthographic(float width, float height, float fNear, float fFar)
    {
        float nearRight = width * 0.5f;
        float farRight = nearRight;
        float nearUp = height * 0.5f;
        float farUp = nearUp;

        Create(nearRight, farRight, nearUp, farUp, fNear, fFar);
    }

    void Frustum::CreatePerspective(float aspect, float fov, float fNear, float fFar)
    {
        float tanFovo2 = (float)tan(fov / 2.0);
        float nearRight = fNear * tanFovo2;// * aspect;
        float farRight = fFar * tanFovo2;// * aspect;
        float nearUp = fNear * tanFovo2 * aspect;
        float farUp = fFar * tanFovo2 * aspect;

        Create(nearRight, farRight, nearUp, farUp, fNear, fFar);
    }

    void Frustum::Create(float nearRight, float farRight, float nearUp, float farUp, float fNear, float fFar)
    {
        m_Initpoints[rightUpNear] = util::Vec3(nearRight, nearUp, fNear);
        m_Initpoints[rightDownNear] = util::Vec3(nearRight, -nearUp, fNear);
        m_Initpoints[leftDownNear] = util::Vec3(-nearRight, -nearUp, fNear);
        m_Initpoints[leftUpNear] = util::Vec3(-nearRight, nearUp, fNear);

        m_Initpoints[rightUpFar] = util::Vec3(farRight, farUp, fFar);
        m_Initpoints[rightDownFar] = util::Vec3(farRight, -farUp, fFar);
        m_Initpoints[leftDownFar] = util::Vec3(-farRight, -farUp, fFar);
        m_Initpoints[leftUpFar] = util::Vec3(-farRight, farUp, fFar);

        m_points[rightUpNear] = util::Vec3(nearRight, nearUp, fNear);
        m_points[rightDownNear] = util::Vec3(nearRight, -nearUp, fNear);
        m_points[leftDownNear] = util::Vec3(-nearRight, -nearUp, fNear);
        m_points[leftUpNear] = util::Vec3(-nearRight, nearUp, fNear);

        m_points[rightUpFar] = util::Vec3(farRight, farUp, fFar);
        m_points[rightDownFar] = util::Vec3(farRight, -farUp, fFar);
        m_points[leftDownFar] = util::Vec3(-farRight, -farUp, fFar);
        m_points[leftUpFar] = util::Vec3(-farRight, farUp, fFar);

        CreatePlanes();
    }

    const util::Vec3* Frustum::GetPoints(void) const
    {
        return m_points;
    }

    bool Frustum::IsInside(const util::Vec3& point) const
    {
        return IsInside(point, 0);
    }

    bool Frustum::IsInside(const util::Vec3& point, float radius) const
    {
        for(uint i = 0; i < 6; ++i)
        {
            if(!m_planes[i].IsInside(point, radius))
            {
                return false;
            }
        }
        return true;
    }

    bool Frustum::IsInside(const util::AxisAlignedBB& aabb) const
    {
        for(uint i = 0; i < 8; ++i)
        {
            if(IsInside(aabb.GetPoint(i)))
            {
               return true;
            }
        }
        return false;
    }

    void Frustum::Transform(const util::Mat4& mat)
    {
        for(UCHAR i = 0; i < 8; ++i)
        {
            m_points[i] = util::Mat4::Transform(mat, m_Initpoints[i]);
        }

        CreatePlanes();
    }

    void Frustum::CreatePlanes(void)
    {
        m_planes[left].Init(m_points[leftDownNear], m_points[leftDownFar], m_points[leftUpFar]);
        m_planes[right].Init(m_points[rightDownNear], m_points[rightUpFar], m_points[rightDownFar]);
        m_planes[up].Init(m_points[leftUpNear], m_points[leftUpFar], m_points[rightUpFar]);
        m_planes[down].Init(m_points[leftDownNear], m_points[rightDownNear], m_points[leftDownFar]);
        m_planes[front].Init(m_points[leftDownFar], m_points[rightDownFar], m_points[leftUpFar]);
        m_planes[back].Init(m_points[rightDownNear], m_points[leftDownNear], m_points[leftUpNear]);
    }

    //**************PointLightFrustum

    void PointLightFrustum::SetMatrices(util::Mat4 mats[6])
    {
        for(UCHAR i = 0; i < 6; ++i)
        {
            m_mats[i] = mats[i];
        }
    }

    void PointLightFrustum::CreatePerspective(float aspect, float fov, float fNear, float fFar)
    {
        for(UCHAR i = 0; i < 6; ++i)
        {
            m_frustums[i].CreatePerspective(aspect, fov, fNear, fFar);
        }
    }

    bool PointLightFrustum::IsInside(const util::Vec3& point, float radius) const
    {
        /*for(UCHAR i = 0; i < 6; ++i)
        {
            if(m_frustums[i].IsInside(point, radius))
            {
                return TRUE;
            }
        } */
        return m_radius + 2 * radius >= (m_position - point).Length();
    }

    void PointLightFrustum::SetParams(float radius, const util::Vec3& pos)
    {
        m_radius = radius;
        m_position = pos;
    }

    bool PointLightFrustum::IsInside(const util::AxisAlignedBB& aabb) const
    {
        LOG_CRITICAL_ERROR("fix me");
        for(UCHAR i = 0; i < 6; ++i)
        {
            if(m_frustums[i].IsInside(aabb))
            {
                return true;
            }
        }
        return false;
    }

    void PointLightFrustum::Transform(const util::Mat4& mat)
    {
        for(UCHAR i = 0; i < 6; ++i)
        {
            m_frustums[i].Transform(mat);
        }
    }
}
