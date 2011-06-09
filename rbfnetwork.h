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
	
		RbfNetwork(unsigned int input_size, unsigned int output_size, vt learning_rate, vt sigma);
		RbfNetwork(std::string filename);
		
		~RbfNetwork();
		void add_kernel(const Vector& center);
		Vector first_layer_output(const Vector& input) const;
		Vector output(const Vector& input) const;
		Matrix first_layer_output(const Matrix& inputs) const ;
		Matrix output(const Matrix& input) const;
		boost::tuple<RbfNetwork::Vector, vt> output_conf(const Vector& input) const ;
		boost::tuples::tuple<RbfNetwork::Matrix, RbfNetwork::Vector> output_conf(const Matrix& input) const;
		Matrix lsqtrain(const Matrix& input,const Matrix& output);
		Vector gdtrain(const Vector& input, const Vector& target);
		Vector gdtrain(const Matrix& input, const Matrix& target);
		boost::tuples::tuple<RbfNetwork::Vector, vt, unsigned int> output_conf_index(const Vector& input) const;
		void remove_kernels(const std::set<unsigned int >& index);
		void select_random_kernels(const Matrix& input, unsigned int size);
		unsigned int remove_unused_nodes(Matrix input, vt thr);
		

		const Matrix& weights() const
		{
			return m_weights;
		}

		vt minDistFromKernel(const Vector& vec) const;
		unsigned int num_kernels() const;

		vt sigma() const
		{
			return m_sigma;
		}

		void setSigma(vt theValue)
		{
			m_sigma = theValue;
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

		void setAddkernels ( bool theValue )
		{
			m_addkernels = theValue;
		}
	

		bool addkernels() const
		{
			return m_addkernels;
		}

		bool setKernels ( const Matrix& theValue )
		{
			//Warning: if the kernel matrix and weights
			//matrix disagree, the latter will be reset
			//and the function will return false
			
			m_kernels = theValue;
			
			if (m_kernels.size1() != m_weights.size1()) {
				unsigned int size = m_kernels.size1();
#ifdef CONSTANT
				m_weights = boost::numeric::ublas::zero_matrix<Matrix::value_type>(1+size,m_output_size);
#else
				m_weights = boost::numeric::ublas::zero_matrix<Matrix::value_type>(size,m_output_size);
#endif
				m_oldw = boost::numeric::ublas::zero_matrix<Matrix::value_type>(m_weights.size1(),m_weights.size2());
				m_oldDerr = boost::numeric::ublas::zero_matrix<Matrix::value_type>(m_weights.size1(),m_weights.size2());
				std::cerr<<"The new kernel matrix has a different size. The weights are reset"<<std::endl;
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
			m_oldw = boost::numeric::ublas::zero_matrix<Matrix::value_type>(m_weights.size1(),m_weights.size2());
			m_oldDerr = boost::numeric::ublas::zero_matrix<Matrix::value_type>(m_weights.size1(),m_weights.size2());
			if (m_kernels.size1() != m_weights.size1()) {
				unsigned int size = m_weights.size1();
				m_kernels = boost::numeric::ublas::zero_matrix<Matrix::value_type>(size,m_input_size);
				std::cerr<<"The new weights matrix has a different size. The kernels are reset"<<std::endl;
				return false;
			}
			else {
				return true;
			}
		}

	protected:
		unsigned int m_input_size;
		unsigned int m_output_size;
		Vector m_learning_rate;
		vt m_sigma;
		Matrix m_kernels;
		Matrix m_weights;
		Matrix m_oldw;
		Matrix m_oldDerr;
		vt m_lveta;
		bool m_addkernels;
		vt m_nimin;
		bool m_firsttrain;
		
	protected:
		RbfNetwork() {/*this is here just for serialization... it does nothing, so don't use it!*/};
		void init_weights(vt a, vt b,unsigned int weightnumber);
		template<class Archive> void serialize(Archive & ar, const unsigned int version) {
			ar & m_firsttrain;
			ar & m_addkernels;
			ar & m_input_size;
			ar & m_learning_rate;
			ar & m_lveta;
			ar & m_output_size;
			ar & m_sigma;
			ar & m_lveta;
			ar & m_nimin;
			ar & m_kernels;
			ar & m_weights;	
			ar & m_oldDerr;
			ar & m_oldw;
		};
		vt armijo(const Vector& Derr, const Vector& oldDerr,const Vector& w, const Vector& oldw, const Vector& kernels_outputs, vt target) ;
};

#endif

