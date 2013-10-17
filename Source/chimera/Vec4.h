#pragma once
#include "stdafx.h"
using namespace DirectX;

namespace chimera
{
    namespace util 
    {
        class Vec4
        {
        public:
            XMFLOAT4 m_v;

            CM_INLINE Vec4() 
            {
                m_v = GetFromVector4(XMVectorZero());
            }

            CM_INLINE Vec4(FLOAT x, FLOAT y, FLOAT z, FLOAT w) 
            {
                this->m_v.x = x;
                this->m_v.y = y;
                this->m_v.z = z;
                this->m_v.w = w;
            }

            CM_INLINE Vec4(CONST Vec4& v)
            {
                this->m_v = v.m_v;
            }

            CM_INLINE Vec4(CONST XMFLOAT4& v) 
            {
                this->m_v = v;
            }

            CM_INLINE FLOAT GetX() CONST {
                return this->m_v.x;
            }

            CM_INLINE FLOAT GetY() CONST 
            {
                return this->m_v.y;
            }

            CM_INLINE FLOAT GetZ() CONST 
            {
                return this->m_v.z;
            }

            CM_INLINE FLOAT GetW() CONST 
            {
                return this->m_v.w;
            }

            CM_INLINE VOID SetX(FLOAT x) 
            {
                this->m_v.x = x;
            }

            CM_INLINE FLOAT GetAxis(UINT axis) CONST
            {
                switch(axis)
                {
                case 0 : return x;
                case 1 : return y;
                case 2 : return z;
                case 3 : return w;
                }
                return 0;
            }

            CM_INLINE VOID SetY(FLOAT y)
            {
                this->m_v.y = y;
            }

            CM_INLINE VOID SetZ(FLOAT z) 
            {
                this->m_v.z = z;
            }

            CM_INLINE VOID SetW(FLOAT w) 
            {
                this->m_v.w = w;
            }

            CM_INLINE VOID Set(FLOAT x, FLOAT y, FLOAT z, FLOAT w) 
            {
                this->m_v.x = x;
                this->m_v.y = y;
                this->m_v.z = z;
                this->m_v.w = w;
            }

            CM_INLINE VOID Add(Vec4& v)
            {
                this->m_v.x += v.x;
                this->m_v.y += v.y;
                this->m_v.z += v.z;
                this->m_v.w += v.w;
            }

            CM_INLINE VOID Add(FLOAT x, FLOAT y, FLOAT z, FLOAT w) 
            {
                this->m_v.x += x;
                this->m_v.y += y;
                this->m_v.z += z;
                this->m_v.w += w;
            }

            CM_INLINE VOID Sub(FLOAT x, FLOAT y, FLOAT z, FLOAT w) 
            {
                this->m_v.x -= x;
                this->m_v.y -= y;
                this->m_v.z -= z;
                this->m_v.w -= w;
            }

            CM_INLINE FLOAT Length() 
            {
                  XMVECTOR tmp = XMVector4Length(XMLoadFloat4(&m_v));
                  return GetFromVector4(tmp).x;
            }

            CM_INLINE Vec4 Normalize() 
            {
                XMVECTOR tmp = XMVector4Normalize(XMLoadFloat4(&m_v));
                m_v = GetFromVector4(tmp);
                return *this;
            }

            CM_INLINE VOID Homogenize(VOID)
            {
                if(m_v.w != 0)
                {
                    m_v.x = m_v.x / m_v.w;
                    m_v.y = m_v.y / m_v.w;
                    m_v.z = m_v.z / m_v.w;
                    m_v.w = 1;
                }
            }

            CM_INLINE FLOAT Dot(CONST Vec4& v) 
            {
                return Vec4::GetDot(*this, v);
            }

            CM_INLINE VOID Print(VOID) CONST 
            {
                DEBUG_OUT_A("Vec4 (%f, %f, %f, %f)\n", this->m_v.x, this->m_v.y, this->m_v.z, this->m_v.w);
            }

            CM_INLINE static float GetDot(CONST Vec4& v0, CONST Vec4& v1) 
            {
                  XMVECTOR m_v = XMVector4Dot(XMLoadFloat4(&v0.m_v), XMLoadFloat4(&v0.m_v));
                  return GetFromVector4(m_v).x;
            }

            CM_INLINE static Vec4 GetNormalize(CONST Vec4& vec) 
            {
                  XMVECTOR m_v = XMVector4Normalize(XMLoadFloat4(&vec.m_v));
                  return Vec4(GetFromVector4(m_v));
            }

             /*CM_INLINE static XMVECTOR GetFromFloat4(XMFLOAT4& v) {
                  return XMLoadFloat4(&v);
             } */

             CM_INLINE static XMFLOAT4 GetFromVector4(CONST XMVECTOR& v) 
             {
                  XMFLOAT4 m_v;
                  XMStoreFloat4(&m_v, v);
                  return m_v;
             }

            __declspec(property(get = GetX, put = SetX)) float x;
            __declspec(property(get = GetY, put = SetY)) float y;
            __declspec(property(get = GetZ, put = SetZ)) float z;
            __declspec(property(get = GetW, put = SetW)) float w;

            __declspec(property(get = GetX, put = SetX)) float r;
            __declspec(property(get = GetY, put = SetY)) float g;
            __declspec(property(get = GetZ, put = SetZ)) float b;
            __declspec(property(get = GetW, put = SetW)) float a;

            ~Vec4() { }
        };
    };
};