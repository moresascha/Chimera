#include "AxisAlignedBB.h"
#include <limits>
namespace chimera
{
    namespace util 
    {
        AxisAlignedBB::AxisAlignedBB(void)
        {
            Clear();
        }

        AxisAlignedBB::AxisAlignedBB(const AxisAlignedBB& aabb)
        {
            Clear();
            for(UCHAR i = 0; i < 12; ++i)
            {
                AddPoint(aabb.GetPoint(i));
            }
            Construct();
        }

        void AxisAlignedBB::Clear(void)
        {
            m_max.Set(-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity());
            m_min.Set(std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity());
        }

        void AxisAlignedBB::AddPoint(const util::Vec3& point)
        {
            m_min.x = min(point.x, m_min.x);
            m_min.y = min(point.y, m_min.y);
            m_min.z = min(point.z, m_min.z);

            m_max.x = max(point.x, m_max.x);
            m_max.y = max(point.y, m_max.y);
            m_max.z = max(point.z, m_max.z);
        }

        void AxisAlignedBB::Construct(void)
        {
            for(uint i = 0; i < 2; ++i)
            {
                for(uint j = 0; j < 2; ++j)
                {
                    for(uint k = 0; k < 2; ++k)
                    {
                        uint pos = 4 * i + 2 * j + k;
                        util::Vec3& p = m_Tpoints[pos];
                        p.x = GetX(i);
                        p.y = GetY(j);
                        p.z = GetZ(k);
                        m_points[pos] = p;
                    }
                }
            }

            //buttom
            m_lines[0] = util::Line(m_min, util::Vec3(m_min.x, m_min.y, m_max.z));
            m_lines[1] = util::Line(util::Vec3(m_min.x, m_min.y, m_max.z), util::Vec3(m_max.x, m_min.y, m_max.z));
            m_lines[2] = util::Line(util::Vec3(m_max.x, m_min.y, m_max.z), util::Vec3(m_max.x, m_min.y, m_min.z));
            m_lines[3] = util::Line(util::Vec3(m_max.x, m_min.y, m_min.z), m_min);

            //middle
            m_lines[4] = util::Line(m_min, util::Vec3(m_min.x, m_max.y, m_min.z));
            m_lines[5] = util::Line(util::Vec3(m_min.x, m_min.y, m_max.z), util::Vec3(m_min.x, m_max.y, m_max.z));
            m_lines[6] = util::Line(util::Vec3(m_max.x, m_min.y, m_max.z), util::Vec3(m_max.x, m_max.y, m_max.z));
            m_lines[7] = util::Line(util::Vec3(m_max.x, m_min.y, m_min.z), util::Vec3(m_max.x, m_max.y, m_min.z));

            //top
            m_lines[8] = util::Line(util::Vec3(m_min.x, m_max.y, m_min.z), util::Vec3(m_min.x, m_max.y, m_max.z));
            m_lines[9] = util::Line(util::Vec3(m_min.x, m_max.y, m_max.z), util::Vec3(m_max.x, m_max.y, m_max.z));
            m_lines[10] = util::Line(util::Vec3(m_max.x, m_max.y, m_max.z), util::Vec3(m_max.x, m_max.y, m_min.z));
            m_lines[11] = util::Line(util::Vec3(m_max.x, m_max.y, m_min.z), util::Vec3(m_min.x, m_max.y, m_min.z));

            /*
            for(UINT i = 0; i < 12; ++i)
            {
                m_lines[i].Print();
            } */

            util::Vec3 m = m_max - m_min;
            m.Scale(0.5f);
            m_mid = m_min + m;
            m_radius = m.Length();
        }

        const util::Vec3& AxisAlignedBB::GetMin(void) const
        {
            return m_min;
        }

        const util::Vec3& AxisAlignedBB::GetMax(void) const
        {
            return m_max;
        }

        const util::Vec3& AxisAlignedBB::GetMiddle(void) const
        {
            return m_mid;
        }

        float AxisAlignedBB::GetMaxAxis(UCHAR axis) const
        {
            switch(axis)
            {
            case 0 : return m_max.x; break;
            case 1 : return m_max.y; break;
            case 2 : return m_max.z; break;
            default : return 0;
            }
        }

        float AxisAlignedBB::GetMinAxis(UCHAR axis) const
        {
            switch(axis)
            {
            case 0 : return m_min.x; break;
            case 1 : return m_min.y; break;
            case 2 : return m_min.z; break;
            default : return 0;
            }
        }

        float AxisAlignedBB::GetX(UCHAR minMax) const
        {
            switch(minMax)
            {
            case 0 : return GetMinAxis(0);
            case 1 : return GetMaxAxis(0);
            default : return 0;
            }
        }

        float AxisAlignedBB::GetY(UCHAR minMax) const
        {
            switch(minMax)
            {
            case 0 : return GetMinAxis(1);
            case 1 : return GetMaxAxis(1);
            default : return 0;
            }
        }

        float AxisAlignedBB::GetZ(UCHAR minMax) const
        {
            switch(minMax)
            {
            case 0 : return GetMinAxis(2);
            case 1 : return GetMaxAxis(2);
            default : return 0;
            }
        }

        float AxisAlignedBB::GetRadius(void) const
        {
            return m_radius;
        }

        bool AxisAlignedBB::IsInside(const util::Vec3& pos) const
        {
            return m_min.x <= pos.x && m_max.x >=pos.x &&
                m_min.y <= pos.y && m_max.y >=pos.y &&
                m_min.z <= pos.z && m_max.z >=pos.z;
        }

        const util::Vec3& AxisAlignedBB::GetPoint(UCHAR i) const
        {
            return m_Tpoints[i];
        }

        const util::Line& AxisAlignedBB::GetLine(UCHAR pos) const
        {
            return m_lines[pos];
        }

        void AxisAlignedBB::SetTransform(util::Mat4& mat)
        {
            for(uint i = 0; i < 12; ++i)
            {
                if(i < 8)
                {
                    m_Tpoints[i] = util::Mat4::Transform(mat, m_points[i]);
                }
                m_lines[i].SetTransform(mat);
            }
            m_min = util::Mat4::Transform(mat, m_min);
            m_max = util::Mat4::Transform(mat, m_max);
            m_mid = m_min + (m_max - m_min) * 0.5;
        }

        AxisAlignedBB::~AxisAlignedBB(void)
        {
        }
    }
}