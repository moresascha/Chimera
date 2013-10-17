#include "Frustum.h"
#include "Line.h"
#include <limits>

CONST UCHAR INS = 0 << 1;
CONST UCHAR LEFT = 1 << 1;
CONST UCHAR RIGHT = 1 << 2;
CONST UCHAR UP = 1 << 3;
CONST UCHAR DOWN = 1 << 4;
CONST UCHAR FRONT = 1 << 5;
CONST UCHAR BACK = 1 << 6;

namespace chimera
{

    VOID Frustum::CreateOrthographicOffCenter(FLOAT l, FLOAT r, FLOAT b, FLOAT t, FLOAT fNear, FLOAT fFar)
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

    VOID Frustum::CreateOrthographic(FLOAT width, FLOAT height, FLOAT fNear, FLOAT fFar)
    {
        FLOAT nearRight = width * 0.5f;
        FLOAT farRight = nearRight;
        FLOAT nearUp = height * 0.5f;
        FLOAT farUp = nearUp;

        Create(nearRight, farRight, nearUp, farUp, fNear, fFar);
    }

    VOID Frustum::CreatePerspective(FLOAT aspect, FLOAT fov, FLOAT fNear, FLOAT fFar)
    {
        FLOAT tanFovo2 = (FLOAT)tan(fov / 2.0);
        FLOAT nearRight = fNear * tanFovo2;// * aspect;
        FLOAT farRight = fFar * tanFovo2;// * aspect;
        FLOAT nearUp = fNear * tanFovo2 * aspect;
        FLOAT farUp = fFar * tanFovo2 * aspect;

        Create(nearRight, farRight, nearUp, farUp, fNear, fFar);
    }

    VOID Frustum::Create(FLOAT nearRight, FLOAT farRight, FLOAT nearUp, FLOAT farUp, FLOAT fNear, FLOAT fFar)
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

    CONST util::Vec3* Frustum::GetPoints(VOID) CONST
    {
        return m_points;
    }

    BOOL Frustum::IsInside(CONST util::Vec3& point) CONST
    {
        return IsInside(point, 0);
    }

    BOOL Frustum::IsInside(CONST util::Vec3& point, FLOAT radius) CONST
    {
        for(UINT i = 0; i < 6; ++i)
        {
            if(!m_planes[i].IsInside(point, radius))
            {
                return FALSE;
            }
        }
        return TRUE;
    }

    BOOL Frustum::IsInside(CONST util::AxisAlignedBB& aabb) CONST
    {
        for(UINT i = 0; i < 8; ++i)
        {
            if(IsInside(aabb.GetPoint(i)))
            {
               return TRUE;
            }
        }
        return FALSE;
    }

    VOID Frustum::Transform(CONST util::Mat4& mat)
    {
        for(UCHAR i = 0; i < 8; ++i)
        {
            m_points[i] = util::Mat4::Transform(mat, m_Initpoints[i]);
        }

        CreatePlanes();
    }

    VOID Frustum::CreatePlanes(VOID)
    {
        m_planes[left].Init(m_points[leftDownNear], m_points[leftDownFar], m_points[leftUpFar]);
        m_planes[right].Init(m_points[rightDownNear], m_points[rightUpFar], m_points[rightDownFar]);
        m_planes[up].Init(m_points[leftUpNear], m_points[leftUpFar], m_points[rightUpFar]);
        m_planes[down].Init(m_points[leftDownNear], m_points[rightDownNear], m_points[leftDownFar]);
        m_planes[front].Init(m_points[leftDownFar], m_points[rightDownFar], m_points[leftUpFar]);
        m_planes[back].Init(m_points[rightDownNear], m_points[leftDownNear], m_points[leftUpNear]);
    }

    //**************PointLightFrustum

    VOID PointLightFrustum::SetMatrices(util::Mat4 mats[6])
    {
        for(UCHAR i = 0; i < 6; ++i)
        {
            m_mats[i] = mats[i];
        }
    }

    VOID PointLightFrustum::CreatePerspective(FLOAT aspect, FLOAT fov, FLOAT fNear, FLOAT fFar)
    {
        for(UCHAR i = 0; i < 6; ++i)
        {
            m_frustums[i].CreatePerspective(aspect, fov, fNear, fFar);
        }
    }

    BOOL PointLightFrustum::IsInside(CONST util::Vec3& point, FLOAT radius) CONST
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

    VOID PointLightFrustum::SetParams(FLOAT radius, CONST util::Vec3& pos)
    {
        m_radius = radius;
        m_position = pos;
    }

    BOOL PointLightFrustum::IsInside(CONST util::AxisAlignedBB& aabb) CONST
    {
        LOG_CRITICAL_ERROR("fix me");
        for(UCHAR i = 0; i < 6; ++i)
        {
            if(m_frustums[i].IsInside(aabb))
            {
                return TRUE;
            }
        }
        return FALSE;
    }

    VOID PointLightFrustum::Transform(CONST util::Mat4& mat)
    {
        for(UCHAR i = 0; i < 6; ++i)
        {
            m_frustums[i].Transform(mat);
        }
    }
}
