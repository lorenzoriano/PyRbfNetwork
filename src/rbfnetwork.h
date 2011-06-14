//
// C++ Interface: rbfmonetwork
//
// Description: 
//
//
// Author:  <>, (C) 2009
//
// Copyright: See COPYING file that comes with this distribution
//
//
#ifndef RBFMONETWORK_H
#define RBFMONETWORK_H

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>
#include <boost/tuple/tuple.hpp>
#include <vector>
#include <string>
#include <utility>
#include <cstring>
#include <iosfwd>
#include <sstream>
#include <set>

#include <exception>

#define CONSTANT

struct rbfn_value_exception : std::exception
{
    rbfn_value_exception(std::string str) {
	msg = str;
    }

    const char* what() const throw() {
	return msg.c_str();
    }

    virtual ~rbfn_value_exception() throw() {

    }

    private:
	std::string msg;
};

template<class T> boost::numeric::ublas::vector<typename T::value_type> sum_rows(T m) {

	using namespace boost::numeric::ublas;
	vector<typename T::value_type> res(m.size1());
	for (unsigned int i=0; i<m.size1(); i++)
		res(i) = sum(row(m,i));
	return res;

};

template<class E, class T, class ME> void write_down(std::basic_ostream<E, T> &os, const boost::numeric::ublas::matrix_expression<ME> &m) { 
	
	typedef typename ME::size_type size_type;
	size_type size1 = m ().size1 ();
	size_type size2 = m ().size2 ();
	std::basic_ostringstream<E, T, std::allocator<E> > s;
	s.flags (os.flags ());
	s.imbue (os.getloc ());
	s.precision (os.precision ());
	
	for (size_type i =0; i<size1; i++) {
		for (size_type j=0; j<size2; j++) {
			s<<m()(i,j);
			if (j < size2-1)
				s<<", ";
		}
		if (i < size1-1)
			s<<"\n";
	}
	os << s.str ().c_str ();
}

template<class E, class T, class VE> void write_down (std::basic_ostream<E, T> &os, const boost::numeric::ublas::vector_expression<VE> &v) {
	
	typedef typename VE::size_type size_type;
	size_type size = v ().size ();
	std::basic_ostringstream<E, T, std::allocator<E> > s;
	s.flags (os.flags ());
	s.imbue (os.getloc ());
	s.precision (os.precision ());
	
	for (size_type i = 0; i < size; ++ i) {
		s << v () (i);
		if (i<size-1)
			s<<", ";
	}
	os << s.str ().c_str ();
}


class RbfNetwork{
	friend class boost::serialization::access;

	public:
		typedef double vt;
		typedef boost::numeric::ublas::matrix<vt> Matrix;
		typedef boost::numeric::ublas::vector<vt> Vector;
	
		RbfNetwork(unsigned int input_size, unsigned int output_size, vt sigma);
		RbfNetwork(std::string filename);
		
		~RbfNetwork();
		Vector first_layer_output(const Vector& input) const;
		Vector output(const Vector& input) const;
		Matrix first_layer_output(const Matrix& inputs) const ;
		Matrix output(const Matrix& input) const;
		boost::tuple<RbfNetwork::Vector, vt> output_conf(const Vector& input) const ;
		boost::tuples::tuple<RbfNetwork::Matrix, RbfNetwork::Vector> output_conf(const Matrix& input) const;
		Matrix lsqtrain(const Matrix& input,const Matrix& output);
		void select_random_kernels(const Matrix& input, unsigned int size);
		

		const Matrix& weights() const
		{
			return m_weights;
		}

		unsigned int num_kernels() const;

		vt sigma() const
		{
			return m_sigma;
		}

		void setSigma(vt theValue)
		{
			m_sigma = theValue * theValue;
		}

		const Matrix& kernels() const
		{
			return m_kernels;
		}

		unsigned int input_size() const
		{
			return m_input_size;
		}

		unsigned int output_size() const
		{
			return m_output_size;
		}
		void save(std::string filename) const;


		bool setKernels ( const Matrix& theValue )
		{
			//Warning: if the kernel matrix and weights
			//matrix disagree, the latter will be reset
			//and the function will return false
			
			m_kernels = theValue;

			if (m_kernels.size1() != m_weights.size1() - 1) {
				unsigned int size = m_kernels.size1() + 1;
				m_weights = boost::numeric::ublas::zero_matrix<Matrix::value_type>(size,m_output_size);
//				std::cerr<<"The new kernel matrix has a different size. The weights are reset"<<std::endl;
				return false;
			}
			else {
				return true;
			}
		}
		
		bool setWeights ( const Matrix& theValue )
		{
			//Warning: if the kernel matrix and weights
			//matrix disagree, the former will be reset
			//and the function will return false
			
			m_weights = theValue;
			if (m_kernels.size1() != m_weights.size1()-1) {
				unsigned int size = m_weights.size1() -1;
				
				m_kernels = boost::numeric::ublas::zero_matrix<Matrix::value_type>(size,m_input_size);
//				std::cerr<<"The new weights matrix has a different size. The kernels are reset"<<std::endl;
				return false;
			}
			else {
				return true;
			}
		}

	protected:
		unsigned int m_input_size;
		unsigned int m_output_size;
		vt m_sigma;
		Matrix m_kernels;
		Matrix m_weights;
		
	protected:
		RbfNetwork() {/*this is here just for serialization... it does nothing, so don't use it!*/};
		void init_weights(vt a, vt b,unsigned int weightnumber);
		template<class Archive> void serialize(Archive & ar, const unsigned int version) {
			ar & m_input_size;
			ar & m_output_size;
			ar & m_sigma;
			ar & m_kernels;
			ar & m_weights;	
		};
};

#endif

