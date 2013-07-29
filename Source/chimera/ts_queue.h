#pragma once
#include "stdafx.h"
#include <queue>
#include "Locker.h"

namespace util {

template<class T>
class ts_queue
{
private:
    std::queue<T> m_queue;
    Locker locker;
public:
    ts_queue(VOID) { }

    BOOL empty() CONST {
        return m_queue.empty();
    }

    size_t size() CONST {
        return m_queue.size();
    }

    T front() CONST {
        return m_queue.front();
    }

    T Back() CONST {
        return m_queue.back();
    }

    T pop(VOID) {
        locker.Lock();
        T t = front();
        m_queue.pop();
        locker.Unlock();
        return t;
    }

    VOID push(T t) {
        locker.Lock();
        m_queue.push(t);
        locker.Unlock();
    }

    ~ts_queue(VOID) { }
};

};