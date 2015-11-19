#pragma once
//#include "stdafx.h"
#include "Vec4.h"
namespace chimera
{
    namespace util 
    {
        class Vec3
        {
        public:
            XMFLOAT3 m_v;

            CM_INLINE Vec3(void) 
            {
                m_v = GetFromVector3(XMVectorZero());
            }

            CM_INLINE Vec3(float x, float y, float z) 
            {
                this->m_v.x = x;
                this->m_v.y = y;
                this->m_v.z = z;
            }

            CM_INLINE Vec3(const Vec3& v) 
            {
                this->m_v = v.m_v;
            }

            CM_INLINE Vec3(const XMFLOAT3& v) 
            {
                this->m_v = v;
            }

            CM_INLINE float GetX() const 
            {
                return this->m_v.x;
            }

            CM_INLINE float GetY() const 
            {
                return this->m_v.y;
            }

            CM_INLINE float GetZ() const 
            {
                return this->m_v.z;
            }

            CM_INLINE void SetX(float x) 
            {
                this->m_v.x = x;
            }

            CM_INLINE void SetY(float y)
            {
                this->m_v.y = y;
            }

            CM_INLINE void SetZ(float z) 
            {
                this->m_v.z = z;
            }

            CM_INLINE float GetAxis(uint axis) const
            {
                switch(axis)
                {
                case 0 : return x;
                case 1 : return y;
                case 2 : return z;
                }
                return 0;
            }

            CM_INLINE float Length() 
            {
                XMVECTOR tmp = XMVector3Length(GetFromFloat3(m_v));
                return GetFromVector3(tmp).x;
            }

            CM_INLINE Vec3& Normalize() 
            {
                XMVECTOR tmp = XMVector3Normalize(GetFromFloat3(m_v));
                m_v = GetFromVector3(tmp);
                return *this;
            }

            CM_INLINE float Dot(const Vec3& v) const 
            {
                //return m_v.x*v.x + m_v.y*v.y + m_v.z*v.z;
                return XMVector3Dot(XMLoadFloat3(&this->m_v), XMLoadFloat3(&v.m_v)).m128_f32[0];
                //return Vec3::GetDot(*this, v);
            }

            CM_INLINE void Add(const Vec3& v) 
            {
                this->m_v.x += v.x;
                this->m_v.y += v.y;
                this->m_v.z += v.z;
            }

            CM_INLINE void Add(float x, float y, float z)
            {
                this->m_v.x += x;
                this->m_v.y += y;
                this->m_v.z += z;
            }

            CM_INLINE void Sub(float x, float y, float z) 
            {
                this->m_v.x -= x;
                this->m_v.y -= y;
                this->m_v.z -= z;
            }

            CM_INLINE void Sub(const Vec3& v) 
            {
                this->m_v.x -= v.x;
                this->m_v.y -= v.y;
                this->m_v.z -= v.z;
            }

            CM_INLINE void Mul(const Vec3& v) 
            {
                this->m_v.x *= v.x;
                this->m_v.y *= v.y;
                this->m_v.z *= v.z;
            }

            CM_INLINE void Scale(float s)
            {
                this->m_v.x *= s;
                this->m_v.y *= s;
                this->m_v.z *= s;
            }

            CM_INLINE void Scale(const util::Vec3& vec) 
            {
                this->m_v.x *= vec.x;
                this->m_v.y *= vec.y;
                this->m_v.z *= vec.z;
            }

            CM_INLINE void Set(float x, float y, float z) 
            {
                this->m_v.x = x;
                this->m_v.y = y;
                this->m_v.z = z;
            }

            CM_INLINE void Set(Vec3& set) 
            {
                this->m_v.x = set.x;
                this->m_v.y = set.y;
                this->m_v.z = set.z;
            }

            Vec3 operator-(const Vec3& right) const 
            {
                Vec3 s(*this);
                s.Sub(right);
                return s;
            }

            Vec3 operator+(const Vec3& right) const 
            {
                Vec3 s(*this);
                s.Add(right);
                return s;
            }

            Vec3 operator*(const Vec3& right) const 
            {
                Vec3 s(*this);
                s.Scale(right);
                return s;
            }

            Vec3 operator*(float s) const 
            {
                util::Vec3 v(*this);
                v.Scale(s);
                return v;
            }

            Vec3 operator/(float s) const 
            {
                util::Vec3 v(*this);
                v.Scale(1.0f / s);
                return v;
            }

            Vec3 operator-() const 
            {
                Vec3 v(*this);
                v.Scale(-1.f);
                return v;
            }

            Vec3& operator= (const Vec3& vec)
            {
                m_v.x = vec.m_v.x;
                m_v.y = vec.m_v.y;
                m_v.z = vec.m_v.z;
                return *this;
            }

            CM_INLINE void Print(void) const 
            {
                DEBUG_OUT_A("Vec3 (%f, %f, %f)\n", this->m_v.x, this->m_v.y, this->m_v.z);
            }

            CM_INLINE static float GetDot(const Vec3& v0, const Vec3& v1) 
            {
                return XMVector3Dot(XMLoadFloat3(&v0.m_v), XMLoadFloat3(&v1.m_v)).m128_f32[0];
            }

            CM_INLINE static Vec3 Sub(Vec3& v0, Vec3& v1)
            {
                Vec3 v(v0);
                v.Sub(v1);
                return v;
            }

            CM_INLINE static Vec3 GetNormalize(const Vec3& vec)
            {
                XMVECTOR m_v = XMVector3Normalize(GetFromFloat3(vec.m_v));
                return GetFromVector3(m_v);
            }

            CM_INLINE static Vec3 GetCross(const Vec3& v0, const Vec3 v1) 
            {
                XMVECTOR m_v0 = GetFromFloat3(v0.m_v);
                XMVECTOR m_v1 = GetFromFloat3(v1.m_v);
                return GetFromVector3(XMVector3Cross(m_v0, m_v1));
            }

            CM_INLINE static Vec3 GetCross(const Vec4& v0, const Vec4 v1)
            {
                Vec3 m_v0(v0.m_v.x, v0.m_v.y, v0.m_v.z);
                Vec3 m_v1(v1.m_v.x, v1.m_v.y, v1.m_v.z);
                return Vec3::GetCross(m_v0, m_v1);
            }

            CM_INLINE static util::Vec3 Min(const util::Vec3& p0, const util::Vec3& p1)
            {
                util::Vec3 result;
                result.x = fminf(p0.x, p1.x);
                result.y = fminf(p0.y, p1.y);
                result.z = fminf(p0.z, p1.z);
                return result;
            }

            CM_INLINE static util::Vec3 Max(const util::Vec3& p0, const util::Vec3& p1)
            {
                util::Vec3 result;
                result.x = fmaxf(p0.x, p1.x);
                result.y = fmaxf(p0.y, p1.y);
                result.z = fmaxf(p0.z, p1.z);
                return result;
            }

            CM_INLINE static util::Vec3 lerp(const util::Vec3& p0, const util::Vec3& p1, float t)
            {
                util::Vec3 result;
                result.x = p0.x * (1-t) + t * p1.x;
                result.y = p0.y * (1-t) + t * p1.y;
                result.z = p0.z * (1-t) + t * p1.z;
                return result;
            }

            CM_INLINE static XMVECTOR GetFromFloat3(const XMFLOAT3& v)
            {
                return XMLoadFloat3(&v);
            }

            CM_INLINE static XMFLOAT3 GetFromVector3(const XMVECTOR& v) 
            {
                XMFLOAT3 m_v;
                XMStoreFloat3(&m_v, v);
                return m_v;
            }

            __declspec(property(get = GetX, put = SetX)) float x;
            __declspec(property(get = GetY, put = SetY)) float y;
            __declspec(property(get = GetZ, put = SetZ)) float z;
        };
    }
};
