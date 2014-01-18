#include "Spline.h"

namespace util
{

    UniformBSpline::UniformBSpline(int devisions /* = 10 */) : m_devisions(devisions), m_pSplinePoints(NULL)
    {

    }

    UniformBSpline::UniformBSpline(const UniformBSpline& spline)
    {
        TBD_FOR(spline.m_controlPoints)
        {
            AddPoint(*it);
        }
        m_devisions = spline.m_devisions;
    }

    const std::vector<util::Vec3>& UniformBSpline::GetControlPoints(void) const
    {
        return m_controlPoints;
    }

    void computeVector(float p0, float p1, float p2, float p3, float a[4])
    {
        a[0] = (-p0 + 3 * p1 - 3 * p2 + p3) / 6.0f;
        a[1] = (3 * p0 - 6 * p1 + 3 * p2) / 6.0f;
        a[2] = (-3 * p0 + 3 * p2) / 6.0f;
        a[3] = (p0 + 4 * p1 + p2) / 6.0f;
    }

    void UniformBSpline::SetDivisions(int divs)
    {
        m_devisions = divs;
        //Create();
    }

    void UniformBSpline::Blend(const util::Vec3& p1, const util::Vec3& p2, const util::Vec3& p3, const util::Vec3& p4, int offset)
    {
        float x[4];
        float y[4];
        float z[4];
        computeVector(p1.x, p2.x, p3.x, p4.x, x);
        computeVector(p1.y, p2.y, p3.y, p4.y, y);
        computeVector(p1.z, p2.z, p3.z, p4.z, z);

        for (int i = 0; i < m_devisions; i++)
        { 
            float t = i / (float)(m_devisions);
            m_pSplinePoints[i + offset].x = t * t * t * x[0] + t * t * x[1] + t * x[2] + x[3];
            m_pSplinePoints[i + offset].y = t * t * t * y[0] + t * t * y[1] + t * y[2] + y[3];
            m_pSplinePoints[i + offset].z = t * t * t * z[0] + t * t * z[1] + t * z[2] + z[3];
        }
    }

    void UniformBSpline::AddPoint(const util::Vec3& p)
    {
        m_controlPoints.push_back(p);
    }

    int UniformBSpline::GetPointsCount(void) const
    {
        return (int)(m_devisions * m_controlPoints.size());
    }

    util::Vec3 UniformBSpline::GetIntpolPoint(float time) const
    {
        int l = GetPointsCount();
        int pos = (int)(l * time);

        return m_pSplinePoints[pos];
    }

    void UniformBSpline::Create(void)
    {
        assert(m_controlPoints.size() > 3);

        if(m_pSplinePoints)
        {
            SAFE_ARRAY_DELETE(m_pSplinePoints);
        }

        m_pSplinePoints = new util::Vec3[m_devisions * m_controlPoints.size()];

        int N = (int)m_controlPoints.size();
        
        for(int i = 0; i < m_controlPoints.size(); ++i)
        {
            int start = (int)m_controlPoints.size() - 1 + i;

            int id0 = (start + 0) % N;
            int id1 = (start + 1) % N;
            int id2 = (start + 2) % N;
            int id3 = (start + 3) % N;

            util::Vec3& pmi  = m_controlPoints[id0];
            util::Vec3& pi   = m_controlPoints[id1];
            util::Vec3& pip1 = m_controlPoints[id2];
            util::Vec3& pip2 = m_controlPoints[id3];

            //DEBUG_OUT_A("%d, %d, %d, %d\n", id0, id1, id2, id3);
            Blend(pmi, pi, pip1, pip2, i * m_devisions);
            /*
            for(INT j = 0; j < 4; ++j)
            {
                INT id0 = (start + 0) % N;
                INT id1 = (start + 1) % N;
                INT id2 = (start + 2) % N;
                INT id3 = (start + 3) % N;

                util::Vec3& pmi  = m_controlPoints[id0];
                util::Vec3& pi   = m_controlPoints[id1];
                util::Vec3& pip1 = m_controlPoints[id2];
                util::Vec3& pip2 = m_controlPoints[id3];

                DEBUG_OUT_A("%d, %d, %d, %d\n", id0, id1, id2, id3);
                Blend(pmi, pi, pip1, pip2, 4 * m_devisions * i + m_devisions * j);
                start++;
            } */
        }
    }

    UniformBSpline::~UniformBSpline(void)
    {
        SAFE_ARRAY_DELETE(m_pSplinePoints);
    }
}