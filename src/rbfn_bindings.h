#ifndef __RBFN_BINDINGS__
#define __RBFN_BINDINGS__

#define Py_USING_UNICODE

#include <boost/python.hpp>
#include <arrayobject.h>

#include <string>
#include "rbfnetwork.h"
#include "storage_adaptors.hpp"
#include "normalizer.h"

namespace ublas = boost::numeric::ublas;

PyArrayObject* arrayobject_from_object(PyObject* o) {
	PyArrayObject* array = (PyArrayObject*)PyArray_FROMANY(o,
					NPY_DOUBLE,1,1, NPY_CARRAY);

	//A few checks
	if (array == NULL)
		throw rbfn_value_exception("Error during conversion. Input is not a 1d vector?");
	if (array->nd > 1)
		throw rbfn_value_exception("Wrong number of dimensions");
	return array;
}

PyArrayObject* matrixobject_from_object(PyObject* o)  {
	PyArrayObject* array = (PyArrayObject*)PyArray_FROMANY(o,
					NPY_DOUBLE,2,2, NPY_CARRAY);

	//A few checks
	if (array == NULL)
		throw rbfn_value_exception("Error during conversion. Input is not a 2d matrix?");
	if (array->nd > 2)
		throw rbfn_value_exception("Wrong number of dimensions");

	return array;
}

template<class Vector> Vector vector_from_object(PyObject *o) {
	PyArrayObject* array_o = arrayobject_from_object(o);

	typedef typename Vector::value_type vt;
	vt* data = ( vt*)PyArray_DATA(array_o);
	size_t size = PyArray_DIM(array_o, 0);

	return ublas::make_vector_from_pointer(size, data);

}
template<class Matrix> Matrix matrix_from_object(PyObject *o) {
		PyArrayObject* matrix_o = matrixobject_from_object(o);

		typedef typename Matrix::value_type vt;
		vt* data = (vt*)PyArray_DATA(matrix_o);

		size_t size1 = PyArray_DIM(matrix_o, 0);
		size_t size2 = PyArray_DIM(matrix_o, 1);

		return ublas::make_matrix_from_pointer(size1, size2, data);

}

template<class Vector> PyObject* vector_to_object(Vector vec) {
	typedef typename Vector::value_type vt;

	size_t size = vec.size();

	npy_intp dims[1];
	dims[0] = size;

	PyObject* out = PyArray_SimpleNew(1,dims, NPY_DOUBLE);
	for (unsigned int i=0; i<size; i++) {
		*((vt*)PyArray_GETPTR1(out, i)) =  vec(i);
	}

	return out;
}

template<class Matrix> PyObject* matrix_to_object(const Matrix& mat)  {

	typedef typename Matrix::value_type vt;
	size_t size1 = mat.size1();
	size_t size2 = mat.size2();

//		npy_intp* dims = new npy_intp[2];
	npy_intp dims[2];
	dims[0] = size1;
	dims[1] = size2;


	PyObject* out = PyArray_SimpleNew(2, dims, NPY_DOUBLE);

	for (unsigned int i=0; i<size1; i++) {
		for (unsigned int j=0; j<size2; j++) {
			*((vt*)PyArray_GETPTR2(out, i,j)) = mat(i,j);
		}
	}

	return out;
}


class RBFN_Wrapper : public RbfNetwork {

public:
	RBFN_Wrapper(unsigned int input_size, unsigned int output_size, float learning_rate, float sigma) :
		RbfNetwork(input_size, output_size, learning_rate, sigma) {}

	RBFN_Wrapper(std::string filename) : RbfNetwork(filename) {}

	void init_weights(float a, float b, unsigned int weightnumber) {
		RbfNetwork::init_weights(a, b, weightnumber);
	}

	void select_random_kernels(PyObject* input, unsigned int size) {
		Matrix mat = matrix_from_object<Matrix>(input);
		RbfNetwork::select_random_kernels(mat, size);
	}

	PyObject* output(PyObject* input) const {
		Matrix mat = matrix_from_object<Matrix>(input);
		Matrix matout = RbfNetwork::output(mat);
		return matrix_to_object(matout);
	}

	PyObject* first_layer_output(PyObject* input) const {
		Matrix mat = matrix_from_object<Matrix>(input);
		Matrix matout = RbfNetwork::first_layer_output(mat);
		return matrix_to_object(matout);
	}

	PyObject* lsqtrain(PyObject* input, PyObject* target) {
		Matrix mat_input = matrix_from_object<Matrix>(input);
		Matrix mat_target = matrix_from_object<Matrix>(target);

		Matrix matout = RbfNetwork::lsqtrain(mat_input, mat_target);
		return matrix_to_object(matout);
	}

	PyObject* weights() const {
		return matrix_to_object(RbfNetwork::weights());
	}

	PyObject* kernels() const {
		return matrix_to_object(RbfNetwork::kernels());
	}

	bool set_weights(PyObject* value) {
		Matrix mat_value = matrix_from_object<Matrix>(value);
		return RbfNetwork::setWeights(mat_value);
	}

	bool set_kernels(PyObject* value) {
		Matrix mat_value = matrix_from_object<Matrix>(value);
		return RbfNetwork::setKernels(mat_value);
	}


private:

};

struct rbfnetwork_pickle_suite : boost::python::pickle_suite
{
	static
	boost::python::tuple
	getinitargs(const RBFN_Wrapper& net)
	{
		unsigned int input_size = net.input_size();
		unsigned int output_size = net.output_size();
		float sigma = net.sigma();
		return boost::python::make_tuple(input_size, output_size, 0, sigma);
	}


	static
	boost::python::tuple
	getstate(boost::python::object  net_obj)
	{
		RBFN_Wrapper const& net = boost::python::extract<RBFN_Wrapper const&>(net_obj)();

		std::stringstream s;
		boost::archive::text_oarchive io(s);
		const RBFN_Wrapper* rnet= dynamic_cast<const RBFN_Wrapper*>(&net);
		io << *rnet;
		return boost::python::make_tuple(
				net_obj.attr("__dict__"),
				s.str()
				);
	}

	static
	void
	setstate(boost::python::object  net_obj, boost::python::tuple state)
	{
		RBFN_Wrapper& net = boost::python::extract<RBFN_Wrapper &>(net_obj)();

		boost::python::dict d = boost::python::extract<boost::python::dict>(
				net_obj.attr("__dict__"))();
		d.update(state[0]);

		std::stringstream s;
		s << boost::python::extract<std::string>(state[1])();
		boost::archive::text_iarchive io(s);
		RBFN_Wrapper* rnet= dynamic_cast<RBFN_Wrapper*>(&net);
		io >> *rnet;
	}

	static bool getstate_manages_dict() { return true; }
};

class Normalizer_Wrapper : public Normalizer {

public:

	Normalizer_Wrapper() : Normalizer() {}
	Normalizer_Wrapper(PyObject* min, PyObject* max) :
		Normalizer(vector_from_object<Vector>(min), vector_from_object<Vector>(max)) {}

	void calculate_from_input(PyObject* input) {
		Normalizer::calculate_from_input(matrix_from_object<Matrix>(input));
	}
	PyObject* deNormalize(PyObject* input) {
		Matrix ret = Normalizer::deNormalize(matrix_from_object<Matrix>(input));
		return matrix_to_object(ret);
	}
	PyObject* normalize(PyObject* input) {
		Matrix ret = Normalizer::normalize(matrix_from_object<Matrix>(input));
		return matrix_to_object(ret);
	}

	PyObject* min() {
		return vector_to_object(m_min);
	}

	PyObject* max() {
		return vector_to_object(m_max);
	}


};

struct normalizer_pickle_suite : boost::python::pickle_suite
{
	static
	boost::python::tuple
	getstate(boost::python::object  norm_obj)
	{
		Normalizer_Wrapper const& norm = boost::python::extract<Normalizer_Wrapper const&>(norm_obj)();

		std::stringstream s;
		boost::archive::text_oarchive io(s);
		const Normalizer_Wrapper* rnorm= dynamic_cast<const Normalizer_Wrapper*>(&norm);
		io << *rnorm;
		return boost::python::make_tuple(
				norm_obj.attr("__dict__"),
				s.str()
				);
	}

	static
	void
	setstate(boost::python::object  norm_obj, boost::python::tuple state)
	{
		Normalizer_Wrapper& norm = boost::python::extract<Normalizer_Wrapper &>(norm_obj)();

		boost::python::dict d = boost::python::extract<boost::python::dict>(
				norm_obj.attr("__dict__"))();
		d.update(state[0]);

		std::stringstream s;
		s << boost::python::extract<std::string>(state[1])();
		boost::archive::text_iarchive io(s);
		Normalizer_Wrapper* rnorm= dynamic_cast<Normalizer_Wrapper*>(&norm);
		io >> *rnorm;
	}

	static bool getstate_manages_dict() { return true; }
};


#endif
