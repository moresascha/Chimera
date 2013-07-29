#pragma once

#include "stdafx.h"
#include "Vec3.h"
#include "Plane.h"
#include "Mat4.h"

namespace util
{
    class Line 
    {
    public:
        util::Vec3 m_p[2];
        util::Vec3 m_Tp[2];

    public:
        Line(VOID) {}

        Line(CONST util::Vec3& p0, CONST util::Vec3& p1)
        {
            m_Tp[0] = this->m_p[0] = p0;
            m_Tp[1] = this->m_p[1] = p1;
        }

        Line(FLOAT p00, FLOAT p01, FLOAT p02, FLOAT p10, FLOAT p11, FLOAT p12)
        {
            Line(util::Vec3(p00, p01, p02), util::Vec3(p10, p11, p12));
        }

        Line(CONST Line& line)
        {
            this->m_p[0] = line.m_p[0];
            this->m_p[1] = line.m_p[1];
        }

        BOOL Intersects(CONST util::Plane& p, util::Vec3* upDown = NULL)
        {
            FLOAT d0 = p.GetDistance(m_Tp[0]);
            FLOAT d1 = p.GetDistance(m_Tp[1]);

            BOOL i0 = d0 <= 0 && d1 > 0;
            BOOL i1 = d0 > 0 && d1 <= 0;
            if(i0 || i1)
            {
                if(upDown)
                {
                    if(i0)
                    {
                        upDown[0] = m_Tp[1];
                        upDown[1] = m_Tp[0];
                    }
                    else
                    {
                        upDown[1] = m_Tp[1];
                        upDown[0] = m_Tp[0];
                    }
                }
                return TRUE;
            }
            return FALSE;
        }

        CONST util::Vec3& GetPoint(UCHAR pos) CONST
        {
            return m_Tp[pos];
        }

        VOID SetTransform(util::Mat4& m)
        {
            m_Tp[0] = util::Mat4::Transform(m, m_p[0]);
            m_Tp[1] = util::Mat4::Transform(m, m_p[1]);
        }

        BOOL IsInside(CONST util::Plane p) CONST
        {
            return p.IsInside(m_Tp[0]) && p.IsInside(m_Tp[1]);
        }

        VOID Print(VOID)
        {
            DEBUG_OUT("Line: { ");
            m_Tp[0].Print();
            m_Tp[1].Print();
            DEBUG_OUT("      }");
        }
    };
}