#pragma once
#include "stdafx.h"
#include <sstream>
//using namespace DirectX;

namespace util 
{
    class DLL_EXPORT Vec4
    {
    public:
        XMFLOAT4 m_v;

        Vec4() {
              m_v = GetFromVector4(XMVectorZero());
        }

        Vec4(FLOAT x, FLOAT y, FLOAT z, FLOAT w) {
            this->m_v.x = x;
            this->m_v.y = y;
            this->m_v.z = z;
            this->m_v.w = w;
        }

        Vec4(CONST Vec4& v) {
            this->m_v = v.m_v;
        }

         Vec4(CONST XMFLOAT4& v) {
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

        inline FLOAT GetW() CONST {
            return this->m_v.w;
        }

        inline VOID SetX(FLOAT x) {
            this->m_v.x = x;
        }

        inline FLOAT GetAxis(UINT axis) CONST
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

        inline VOID SetY(FLOAT y) {
            this->m_v.y = y;
        }

        inline VOID SetZ(FLOAT z) {
            this->m_v.z = z;
        }

        inline VOID SetW(FLOAT w) {
            this->m_v.w = w;
        }

        inline VOID Set(FLOAT x, FLOAT y, FLOAT z, FLOAT w) {
            this->m_v.x = x;
            this->m_v.y = y;
            this->m_v.z = z;
            this->m_v.w = w;
        }

        inline VOID Add(Vec4& v) {
            this->m_v.x += v.x;
            this->m_v.y += v.y;
            this->m_v.z += v.z;
            this->m_v.w += v.w;
        }

        inline VOID Add(FLOAT x, FLOAT y, FLOAT z, FLOAT w) {
            this->m_v.x += x;
            this->m_v.y += y;
            this->m_v.z += z;
            this->m_v.w += w;
        }

        inline VOID Sub(FLOAT x, FLOAT y, FLOAT z, FLOAT w) {
            this->m_v.x -= x;
            this->m_v.y -= y;
            this->m_v.z -= z;
            this->m_v.w -= w;
        }

        inline FLOAT Length() {
              XMVECTOR tmp = XMVector4Length(XMLoadFloat4(&m_v));
              return GetFromVector4(tmp).x;
        }

        inline Vec4 Normalize() {
            XMVECTOR tmp = XMVector4Normalize(XMLoadFloat4(&m_v));
              m_v = GetFromVector4(tmp);
            return *this;
        }

        inline VOID Homogenize(VOID)
        {
            if(m_v.w != 0)
            {
                m_v.x = m_v.x / m_v.w;
                m_v.y = m_v.y / m_v.w;
                m_v.z = m_v.z / m_v.w;
                m_v.w = 1;
            }
        }

        inline FLOAT Dot(CONST Vec4& v) {
            return Vec4::GetDot(*this, v);
        }

        inline VOID Print(VOID) CONST 
        {
            DEBUG_OUT_A("Vec4 (%f, %f, %f, %f)", this->m_v.x, this->m_v.y, this->m_v.z, this->m_v.w);
        }

        inline static float GetDot(CONST Vec4& v0, CONST Vec4& v1) {
              XMVECTOR m_v = XMVector4Dot(XMLoadFloat4(&v0.m_v), XMLoadFloat4(&v0.m_v));
              return GetFromVector4(m_v).x;
        }

        inline static Vec4 GetNormalize(CONST Vec4& vec) {
              XMVECTOR m_v = XMVector4Normalize(XMLoadFloat4(&vec.m_v));
              return Vec4(GetFromVector4(m_v));
        }

         /*inline static XMVECTOR GetFromFloat4(XMFLOAT4& v) {
              return XMLoadFloat4(&v);
         } */

         inline static XMFLOAT4 GetFromVector4(XMVECTOR v) {
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

    typedef Vec4 Color;
}

