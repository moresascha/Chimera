#pragma once
#include "stdafx.h"
#include "Vec4.h"
#include <sstream>
//using namespace DirectX;
namespace util 
{
    class DLL_EXPORT Vec3
    {
    public:
        XMFLOAT3 m_v;
        static CONST util::Vec3 X_AXIS;
        static CONST util::Vec3 Y_AXIS;
        static CONST util::Vec3 Z_AXIS;
        Vec3() {
              m_v = GetFromVector3(XMVectorZero());
        }

        Vec3(FLOAT x, FLOAT y, FLOAT z) {
            this->m_v.x = x;
            this->m_v.y = y;
            this->m_v.z = z;
        }

        Vec3(CONST Vec3& v) {
            this->m_v = v.m_v;
        }

         Vec3(CONST XMFLOAT3& v) {
            this->m_v = v;
        }

         inline FLOAT GetX() CONST {
              return this->m_v.x;
         }
     
         inline FLOAT GetY() CONST {
              return this->m_v.y;
         }

         inline FLOAT GetZ() CONST {
              return this->m_v.z;
         }

         inline VOID SetX(FLOAT x) {
              this->m_v.x = x;
         }
     
         inline VOID SetY(FLOAT y) {
              this->m_v.y = y;
         }

         inline VOID SetZ(FLOAT z) {
              this->m_v.z = z;
         }

        inline FLOAT GetAxis(UINT axis) CONST
        {
            switch(axis)
            {
            case 0 : return x;
            case 1 : return y;
            case 2 : return z;
            }
            return 0;
        }

        inline FLOAT Length() {
            XMVECTOR tmp = XMVector3Length(GetFromFloat3(m_v));
            return GetFromVector3(tmp).x;
        }

        inline Vec3& Normalize() {
            XMVECTOR tmp = XMVector3Normalize(GetFromFloat3(m_v));
            m_v = GetFromVector3(tmp);
            return *this;
        }

        inline FLOAT Dot(CONST Vec3& v) CONST {
            //return m_v.x*v.x + m_v.y*v.y + m_v.z*v.z;
            return XMVector3Dot(XMLoadFloat3(&this->m_v), XMLoadFloat3(&v.m_v)).m128_f32[0];
            //return Vec3::GetDot(*this, v);
        }

         inline VOID Add(CONST Vec3& v) {
              this->m_v.x += v.x;
              this->m_v.y += v.y;
              this->m_v.z += v.z;
         }

        inline VOID Add(FLOAT x, FLOAT y, FLOAT z) {
            this->m_v.x += x;
            this->m_v.y += y;
            this->m_v.z += z;
        }

        inline VOID Sub(FLOAT x, FLOAT y, FLOAT z) {
            this->m_v.x -= x;
            this->m_v.y -= y;
            this->m_v.z -= z;
        }

        inline VOID Sub(CONST Vec3& v) {
            this->m_v.x -= v.x;
            this->m_v.y -= v.y;
            this->m_v.z -= v.z;
        }

         inline VOID Mul(CONST Vec3& v) {
              this->m_v.x *= v.x;
              this->m_v.y *= v.y;
              this->m_v.z *= v.z;
         }

         inline VOID Scale(FLOAT s) {
              this->m_v.x *= s;
              this->m_v.y *= s;
              this->m_v.z *= s;
         }

        inline VOID Scale(CONST util::Vec3& vec) {
            this->m_v.x *= vec.x;
            this->m_v.y *= vec.y;
            this->m_v.z *= vec.z;
        }

         inline VOID Set(FLOAT x, FLOAT y, FLOAT z) {
              this->m_v.x = x;
              this->m_v.y = y;
              this->m_v.z = z;
         }

        inline VOID Set(Vec3& set) {
            this->m_v.x = set.x;
            this->m_v.y = set.y;
            this->m_v.z = set.z;
        }

        Vec3 operator-(CONST Vec3& right) CONST {
            Vec3 s(*this);
            s.Sub(right);
            return s;
        }

        Vec3 operator+(CONST Vec3& right) CONST {
            Vec3 s(*this);
            s.Add(right);
            return s;
        }

        Vec3 operator*(CONST Vec3& right) CONST {
            Vec3 s(*this);
            s.Scale(right);
            return s;
        }

        Vec3 operator*(FLOAT s) CONST {
            util::Vec3 v(*this);
            v.Scale(s);
            return v;
        }

        Vec3 operator/(FLOAT s) CONST {
            util::Vec3 v(*this);
            v.Scale(1.0f / s);
            return v;
        }

        Vec3 operator-() CONST {
            Vec3 v(*this);
            v.Scale(-1.f);
            return v;
        }

        Vec3& operator= (CONST Vec3& vec)
        {
            m_v.x = vec.m_v.x;
            m_v.y = vec.m_v.y;
            m_v.z = vec.m_v.z;
            return *this;
        }

        inline VOID Print(VOID) CONST {
            DEBUG_OUT_A("Vec3 (%f, %f, %f)\n", this->m_v.x, this->m_v.y, this->m_v.z);
        }

        inline static float GetDot(CONST Vec3& v0, CONST Vec3& v1) {
              return XMVector3Dot(XMLoadFloat3(&v0.m_v), XMLoadFloat3(&v1.m_v)).m128_f32[0];
        }

        inline static Vec3 Sub(Vec3& v0, Vec3& v1) {
            Vec3 v(v0);
            v.Sub(v1);
            return v;
        }

        inline static Vec3 GetNormalize(CONST Vec3& vec) {
              XMVECTOR m_v = XMVector3Normalize(GetFromFloat3(vec.m_v));
              return GetFromVector3(m_v);
        }

        inline static Vec3 GetCross(CONST Vec3& v0, CONST Vec3 v1) {
            XMVECTOR m_v0 = GetFromFloat3(v0.m_v);
              XMVECTOR m_v1 = GetFromFloat3(v1.m_v);
              return GetFromVector3(XMVector3Cross(m_v0, m_v1));
        }
  
        inline static Vec3 GetCross(CONST Vec4& v0, CONST Vec4 v1) {
            Vec3 m_v0(v0.m_v.x, v0.m_v.y, v0.m_v.z);
              Vec3 m_v1(v1.m_v.x, v1.m_v.y, v1.m_v.z);
              return Vec3::GetCross(m_v0, m_v1);
        }

        inline static util::Vec3 Min(CONST util::Vec3& p0, CONST util::Vec3& p1)
        {
            util::Vec3 result;
            result.x = min(p0.x, p1.x);
            result.y = min(p0.y, p1.y);
            result.z = min(p0.z, p1.z);
            return result;
        }

        inline static util::Vec3 Max(CONST util::Vec3& p0, CONST util::Vec3& p1)
        {
            util::Vec3 result;
            result.x = max(p0.x, p1.x);
            result.y = max(p0.y, p1.y);
            result.z = max(p0.z, p1.z);
            return result;
        }

        inline static util::Vec3 lerp(CONST util::Vec3& p0, CONST util::Vec3& p1, FLOAT t)
        {
            util::Vec3 result;
            result.x = p0.x * (1-t) + t * p1.x;
            result.y = p0.y * (1-t) + t * p1.y;
            result.z = p0.z * (1-t) + t * p1.z;
            return result;
        }

         inline static XMVECTOR GetFromFloat3(XMFLOAT3 v) {
              return XMLoadFloat3(&v);
         }

         inline static XMFLOAT3 GetFromVector3(XMVECTOR v) {
              XMFLOAT3 m_v;
              XMStoreFloat3(&m_v, v);
              return m_v;
         }

        __declspec(property(get = GetX, put = SetX)) float x;
        __declspec(property(get = GetY, put = SetY)) float y;
        __declspec(property(get = GetZ, put = SetZ)) float z;
    };
}

