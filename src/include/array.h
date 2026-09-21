/*
 * array.h
 *
 * Template classes to manage allocations of multi-dimensional arrays
 * in contiguous memory
 *
 *  Created on: 28 May 2018
 *      Author: drw
 */

#ifndef ARRAY_H_
#define ARRAY_H_

#include <fstream>
#include <iostream>

template<typename T>
class Array
{
public:
    T *ptr;
    int num;
public:
    Array(int N, bool init_zero = true) : num(N)
    {
        ptr = new T[N];
        if(init_zero) zero();
    }

    ~Array()
    {
        delete[] ptr;
    }

    T &operator[](int i)
    {
        return ptr[i];
    }

    operator T *()
    {
        return ptr;
    }

    int len()
    {
        return num;
    }

    void zero()
    {
        for(int i = 0; i < num; i++)
            ptr[i] = 0;
    }

    void write(std::ofstream *outfile)
    {
        outfile->write(reinterpret_cast<char *> (ptr), num * sizeof(T));
    }

    void read(std::ifstream *infile)
    {
        infile->read(reinterpret_cast<char *> (ptr), num * sizeof(T));
    }

    template<typename T2>
    void operator/=(Array<T2> &other)
    {
        if(num != other.num)
        {
            std::cerr << "Array ERROR: Cannot divide arrays with different dimensions";
            return;
        }
        for(int i = 0; i < num; i++)
            ptr[i] /= other.ptr[i];
    }

    template<typename T2>
    void operator+=(Array<T2> &other)
    {
        if(num != other.num)
        {
            std::cerr << "Array ERROR: Cannot add arrays with different dimensions";
            return;
        }
        for(int i = 0; i < num; i++)
            ptr[i] += other.ptr[i];
    }

    void operator /= (float other)
    {
        for(int i = 0; i < num; i++)
            ptr[i] /= other;
    }
    void operator /= (double other)
    {
        for(int i = 0; i < num; i++)
            ptr[i] /= other;
    }
    void operator /= (int other)
    {
        for(int i = 0; i < num; i++)
            ptr[i] /= other;
    }
    void operator /= (long other)
    {
        for(int i = 0; i < num; i++)
            ptr[i] /= other;
    }
};

template<typename T>
class Array2D
{
public:
    T **ptr;
    //T* pool;
    int num_x, num_y;
public:
    Array2D(int Nx, int Ny, bool init_zero = true) : num_x(Nx), num_y(Ny)
    {
        ptr = new T *[Nx];
        T *pool = new T[Nx * Ny];
        for(int i = 0; i < Nx; ++i, pool += Ny)
            ptr[i] = pool;

        if(init_zero) zero();
    }

    ~Array2D()
	{
		delete[] ptr[0];
		delete[] ptr;
	}

	T* operator [] (int i)
	{
		return ptr[i];
	}

    operator T **()
    {
        return ptr;
    }

    operator T *()
    {
        return ptr[0];
    }

    int size_x()
    {
        return num_x;
    }

    int size_y()
    {
        return num_y;
    }

    void zero()
    {
        for(int i = 0; i < num_x * num_y; i++)
            ptr[0][i] = 0;
    }

    void write(std::ofstream *outfile)
    {
        outfile->write(reinterpret_cast<char *> (ptr[0]), num_x * num_y * sizeof(T));
	}

	void read(std::ifstream* infile)
	{
		infile->read(reinterpret_cast<char*> (ptr[0]), num_x * num_y * sizeof(T));
	}

	template<typename T2>
	void operator /= (Array2D<T2>& other)
	{
		if(num_x != other.num_x || num_y != other.num_y)
		{
			std::cerr << "Array2D ERROR: Cannot divide arrays with different dimensions";
			return;
		}
		for(int i=0; i<num_x*num_y; i++)
			ptr[0][i] /= other.ptr[0][i];
	}

    void operator /= (float other)
    {
        for(int i=0; i<num_x*num_y; i++)
            ptr[0][i] /= other;
    }
    void operator /= (double other)
    {
        for(int i=0; i<num_x*num_y; i++)
            ptr[0][i] /= other;
    }
    void operator /= (int other)
    {
        for(int i=0; i<num_x*num_y; i++)
            ptr[0][i] /= other;
    }
    void operator /= (long other)
    {
        for(int i=0; i<num_x*num_y; i++)
            ptr[0][i] /= other;
    }

	template<typename T2>
	void operator += (Array2D<T2>& other)
	{
		if(num_x != other.num_x || num_y != other.num_y)
		{
			std::cerr << "Array2D ERROR: Cannot add arrays with different dimensions";
			return;
		}
		for(int i=0; i<num_x*num_y; i++)
			ptr[0][i] += other.ptr[0][i];
	}
};

template <typename T>
class Array3D
{
public:
    T ***ptr;
    T *pool;
    int num_x, num_y, num_z;
public:
    Array3D(int Nx, int Ny, int Nz, bool init_zero = true) : num_x(Nx), num_y(Ny), num_z(Nz)
    {
        ptr = new T **[Nx];
        pool = new T[Nx * Ny * Nz];
        for(int i = 0; i < Nx; ++i)
        {
            ptr[i] = new T *[Ny];
            for(int j = 0; j < Ny; j++)
            {
                ptr[i][j] = pool + i * (Ny * Nz) + j * (Nz);
            }
        }

        if(init_zero) zero();
    }

	~Array3D()
	{
		delete[] ptr[0];
		delete[] ptr;
	}

	T& elem(int i, int j, int k)
    {
	    return ptr[i][j][k];
    }

	T** operator [] (int i)
	{
		return ptr[i];
	}

    operator T **()
    {
        return ptr[0];
    }

    operator T *()
    {
        return ptr[0][0];
    }

    int size_x()
    {
        return num_x;
    }

    int size_y()
    {
        return num_y;
    }

    int size_z()
    {
        return num_z;
    }

    void zero()
    {
        for(int i = 0; i < num_x * num_y * num_z; i++)
            pool[i] = 0;
    }

    void write(std::ofstream *outfile)
    {
        outfile->write(reinterpret_cast<char *> (ptr[0][0]), num_x * num_y * num_z * sizeof(T));
	}

	void read(std::ifstream* infile)
	{
		infile->read(reinterpret_cast<char*> (ptr[0][0]), num_x * num_y * num_z * sizeof(T));
	}

	template<typename T2>
    void operator /= (Array3D<T2>& other)
    {
        if(num_x != other.num_x || num_y != other.num_y || num_z != other.num_z)
        {
            std::cerr << "Array3D ERROR: Cannot divide arrays with different dimensions";
            return;
        }
        for(int i=0; i<num_x*num_y*num_z; i++)
            pool[i] /= other.pool[i];
    }

    void operator /= (float other)
    {
        for(int i=0; i<num_x*num_y*num_z; i++)
            pool[i] /= other;
    }
    void operator /= (double other)
    {
        for(int i=0; i<num_x*num_y*num_z; i++)
            pool[i] /= other;
    }
    void operator /= (int other)
    {
        for(int i=0; i<num_x*num_y*num_z; i++)
            pool[i] /= other;
    }
    void operator /= (long other)
    {
        for(int i=0; i<num_x*num_y*num_z; i++)
            pool[i] /= other;
    }

    template<typename T2>
    void operator += (Array3D<T2>& other)
    {
        if(num_x != other.num_x || num_y != other.num_y || num_z != other.num_z)
        {
            std::cerr << "Array3D ERROR: Cannot divide arrays with different dimensions";
            return;
        }
        for(int i=0; i<num_x*num_y*num_z; i++)
            pool[i] += other.pool[i];
    }
};

#endif /* ARRAY_H_ */
