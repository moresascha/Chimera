#pragma once
#include "stdafx.h"
#include "Vec3.h"

namespace util
{
    class UniformBSpline
    {
    private:
        INT m_devisions;
        std::vector<util::Vec3> m_controlPoints;
        util::Vec3* m_pSplinePoints;

        VOID Blend(CONST util::Vec3& p1, CONST util::Vec3& p2, CONST util::Vec3& p3, CONST util::Vec3& p4, int offset);

    public:

        UniformBSpline(INT devisions = 10);

        UniformBSpline(CONST UniformBSpline& spline);

        VOID AddPoint(CONST util::Vec3& p);

        VOID SetDivisions(INT divs);

        VOID Create(VOID);

        INT GetPointsCount(VOID) CONST;

        util::Vec3 GetIntpolPoint(FLOAT time) CONST;

        CONST std::vector<util::Vec3>& GetControlPoints(VOID) CONST;

        UINT GetDivisions(VOID) CONST { return m_devisions; }

        VOID operator=(CONST UniformBSpline& spline)
        {
            m_devisions = spline.m_devisions;
            TBD_FOR(spline.m_controlPoints)
            {
                AddPoint(*it);
            }
        }

        ~UniformBSpline(VOID);
    };
}