#pragma once

template <class T, size_t h>
class VEB
{
        template <class U, size_t a>
        friend class VEB;

private:
        VEB<T, h-h/2> top;
        VEB<T, h/2> down[(1<<(h-h/2))];

        static size_t low(size_t u)
        {
                return u & ((1ull<<(h/2))-1);
        }
        
        static size_t high(size_t u)
        {
                return u >> (h/2);
        }

        /*
         * @return -1 wenn k gefunden, sonst den Index des Subtrees, in dem weitergesucht werden muss.
         */
        long long psearch(const T& k) const
        {
                int child = top.psearch(k);
                if(child < 0)
                        return -1;
                int child2 = down[child].psearch(k);
                if(child2 < 0)
                        return -1;
                return (1<<(h/2))*child + child2;
        }

public:
        static size_t size()
        {
                return (1<<h) - 1;
        }

        VEB(){}

        VEB(T* ar)
        {
                for(size_t i = 0; i < size(); i++)
                {
                        (*this)[i] = ar[i];
                }
        }

        T& operator [] (size_t index)
        {
                if(low(index) == low((size_t)-1))
                        return top[high(index)];
                return down[high(index)][low(index)];
        }

        bool search(const T& k) const
        {
                return psearch(k) < 0;
        }

};


template <class T>
class VEB<T, 1>
{
        template <class U, size_t a>
        friend class VEB;

private:
        T data;

        int psearch(const T& k) const
        {
                if(k < data)
                        return 0;
                if(k > data)
                        return 1;
                return -1;
        }

public:
        static size_t size()
        {
                return 1;
        }
        
        VEB():data(){}

        VEB(T* ar)
        {
                data = *ar;
        }

        T& operator [] (size_t index)
        {
                return data;
        }

        bool search(const T& k) const
        {
                return psearch(k) < 0;
        }
};