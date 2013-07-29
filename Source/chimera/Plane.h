#pragma once
#include "stdafx.h"
#include "Vec3.h"
namespace util 
{
    class Plane 
    {

    private:
        util::Vec3 m_normal;
        FLOAT m_radius;

    public:
        Plane(VOID) {}

        //CCW
        VOID Init(CONST util::Vec3& p0, CONST util::Vec3& p1, CONST util::Vec3& p2)
        {
            util::Vec3 t0 = p1 - p0;
            util::Vec3 t1 = p2 - p0;
            util::Vec3 cross = util::Vec3::GetCross(t0, t1);
            cross.Normalize();
            FLOAT radius = util::Vec3::GetDot(cross, p0);
            Init(cross, radius);
        }

        VOID Init(CONST util::Vec3& normal, FLOAT radius)
        {
            m_normal = normal;
            m_normal.Normalize();
            m_radius = radius;
        }

        FLOAT GetDistance(CONST util::Vec3& point) CONST
        {
            return point.Dot(m_normal) - m_radius;
        }

        BOOL IsInside(CONST util::Vec3& point) CONST
        {
            return GetDistance(point) <= 0;
        }

        BOOL IsInside(CONST util::Vec3& point, FLOAT radius) CONST
        {
            return GetDistance(point) <= radius;
        }

        util::Vec3 GetNormal(VOID) CONST
        {
            return m_normal;
        }

        FLOAT GetRadius(VOID) CONST
        {
            return m_radius;
        }
    };
}