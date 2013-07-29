#pragma once
#include "stdafx.h"
namespace util 
{
    template <class T>
    class tbdStack
    {
    public:
        std::list<T> m_stack;
        tbdStack(VOID) {}

        T Pop(VOID)
        {
            T t = m_stack.front();
            m_stack.pop_front();
            return t;
        }

        VOID Push(CONST T& t)
        {
            m_stack.push_front(t);
        }

        T Peek(VOID)
        {
            return m_stack.front();
        }

        T Front(VOID)
        {
            return m_stack.front();
        }

        BOOL IsEmpty(VOID) CONST
        {
            return m_stack.empty();
        }

        VOID Clear(VOID)
        {
            m_stack.clear();
        }

        size_t Size(VOID) CONST
        {
            return m_stack.size();
        }

        ~tbdStack(VOID) {}
    };
}