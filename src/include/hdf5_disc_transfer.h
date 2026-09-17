/*
 * disc_transfer.h
 *
 *  Created on: 17 Oct 2013
 *      Author: drw
 */

#ifndef DISC_TRANSFER_H_
#define DISC_TRANSFER_H_

#include "H5Cpp.h"
using namespace H5;

#define GROUPNAME "/TRANSFER"
#define MAPGROUPNAME "/MAP"

class TransferException : public exception
{
	string what_msg;
public:
	TransferException(string msg) : what_msg("DiscTransfer ERROR : " + msg)
	{

	}
	virtual ~TransferException() throw()
	{   }

	virtual const char* what() const throw()
	{
		return what_msg.c_str();
	}
};

template <typename T>
class DiscTransfer
{
public:
	T *t, *g;
	T *N;
	T *map_x, *map_y;
	long *count;
	int num;

	T r0, dr, dr_inside, phi0, dphi, r_max, phi_max;
	T r_isco;
	int Nr, Nphi, Nr_inside;
	bool logbin_r;

	bool area_div;

	T spin, incl;

private:
	void read_attributes(Group& trf_group)
	{
		trf_group.openAttribute("r0").read(PredType::NATIVE_DOUBLE, &r0);
		trf_group.openAttribute("dr").read(PredType::NATIVE_DOUBLE, &dr);
		trf_group.openAttribute("Nr").read(PredType::NATIVE_INT, &Nr);
		trf_group.openAttribute("phi0").read(PredType::NATIVE_DOUBLE, &phi0);
		trf_group.openAttribute("dphi").read(PredType::NATIVE_DOUBLE, &dphi);
		trf_group.openAttribute("Nphi").read(PredType::NATIVE_INT, &Nphi);

		if(trf_group.attrExists("Nr_inside"))
			trf_group.openAttribute("Nr_inside").read(PredType::NATIVE_INT, &Nr_inside);
		else
			Nr_inside = 0;

		if(trf_group.attrExists("dr_inside"))
			trf_group.openAttribute("dr_inside").read(PredType::NATIVE_DOUBLE, &r_isco);
		else
			r_isco = 0;

		if(trf_group.attrExists("r_isco"))
			trf_group.openAttribute("r_isco").read(PredType::NATIVE_DOUBLE, &r_isco);
		else
			r_isco = 0;

		int logbin_r_int;
		trf_group.openAttribute("logbin_r").read(PredType::NATIVE_INT, &logbin_r_int);
		logbin_r = (logbin_r_int == 1);
		int area_div_int;
		trf_group.openAttribute("divarea").read(PredType::NATIVE_INT, &area_div_int);
		area_div = (area_div_int == 1);
	}

	void read_info(H5File& hdffile)
	{
		hdffile.openAttribute("spin").read(PredType::NATIVE_DOUBLE, &spin);
		hdffile.openAttribute("incl").read(PredType::NATIVE_DOUBLE, &incl);
	}

	template <typename array_T>
	int read_data(DataSet& dataset, array_T*& array, PredType data_type)
	{
		DataSpace dataspace = dataset.getSpace();

		// get the dimensions of the dataspace
		int rank = dataspace.getSimpleExtentNdims();
		hsize_t trf_dims[rank];
		int ndims = dataspace.getSimpleExtentDims(trf_dims, NULL);

		// allocate the array to hold the data
		int num = trf_dims[0]*trf_dims[1];
		array = new array_T[num];

		DataSpace memspace(rank, trf_dims);
		dataset.read(array, data_type, memspace, dataspace);

		return num;
	}

	template <typename array_T>
	int read_dataset(Group& group, H5std_string dataset_name, array_T*& array, PredType data_type)
	{
		DataSet dataset = group.openDataSet(dataset_name);
		int num = read_data<T>(dataset, array, data_type);
		dataset.close();

		return num;
	}

	void show_info(H5File& hdffile)
	{
		cout << "Spin = " << spin << endl;
		cout << "Incl = " << incl << endl;
	}


public:
	DiscTransfer(char* filename, bool read_map = false, bool read_count = false)
	{
		cout << "Reading disc transfer function from HDF5 file: " << filename << endl;

		H5File hdffile(filename, H5F_ACC_RDONLY);
		H5std_string group_name(GROUPNAME);
		Group trf_group = hdffile.openGroup(group_name);

		read_info(hdffile);
		read_attributes(trf_group);
		num = read_dataset<T>(trf_group, "TIME", t, PredType::NATIVE_DOUBLE);
		num = read_dataset<T>(trf_group, "REDSHIFT", g, PredType::NATIVE_DOUBLE);
		num = read_dataset<T>(trf_group, "IMAGE_AREA", N, PredType::NATIVE_DOUBLE);

		if(read_map)
		{
			H5std_string map_group_name(MAPGROUPNAME);
			Group map_group = hdffile.openGroup(map_group_name);
			read_dataset<T>(map_group, "MAP_X", map_x, PredType::NATIVE_DOUBLE);
			read_dataset<T>(map_group, "MAP_Y", map_y, PredType::NATIVE_DOUBLE);
		}

		r_max = (logbin_r) ? r0 * pow(dr, Nr-1) : r0 + (Nr-1)*dr;
		phi_max = phi0 + Nphi*dphi;

		show_info(hdffile);
		cout << "r: " << Nr << (char*)((logbin_r) ? " logarithmic bins " : " bins ") << r0 << ':' << dr << ':' << r_max << "   phi: " << Nphi << " bins "<< phi0 << ':' << dphi << ':' << phi_max << endl;
		cout << "Transfer function has " << num << " bins" << endl << endl;
	}


    ~DiscTransfer()
    {
        cout << "Cleaning up disc transfer function" << endl;
        delete[] t;
        delete[] g;
        delete[] N;
    }


    inline int array_index(int ir, int iphi)
    {
        return ir * Nphi + iphi;
    }

    inline int r_index(T r)
    {
        if(Nr_inside > 0)
        {
            if(r < r_isco)
            {
				return (logbin_r) ? static_cast<int>( log(r/r0) / log(dr_inside) ) : static_cast<int>((r - r0)/dr_inside);
			}
			else
			{
				return Nr_inside + ((logbin_r) ? static_cast<int>( log(r/r_isco) / log(dr) ) : static_cast<int>((r - r_isco)/dr));
			}
		}
		else
        {
            return (logbin_r) ? static_cast<int>( log(r / r0) / log(dr)) : static_cast<int>((r - r0) / dr);
        }
    }

    inline int phi_index(T phi)
    {
        return static_cast<int> ((phi - phi0) / dphi);
    }

    inline int find_bin(T r, T phi)
    {
        return array_index(r_index(r), phi_index(phi));
    }

    inline T r_value(int ir)
    {
        return (logbin_r) ? r0 * pow(dr, ir) : r0 + dr * ir;
    }

};


#endif /* DISC_TRANSFER_H_ */
