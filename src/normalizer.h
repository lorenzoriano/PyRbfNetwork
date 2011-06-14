//
// C++ Interface: normalizer
//
// Description: 
//
//
// Author:  <>, (C) 2009
//
// Copyright: See COPYING file that comes with this distribution
//
//
#ifndef NORMALIZER_H
#define NORMALIZER_H

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <algorithm>
#include <fstream>
/**
	@author 
 */
class Normalizer{
	public:
		typedef boost::numeric::ublas::vector<double> Vector;
		typedef boost::numeric::ublas::matrix<double> Matrix;
		
		Normalizer() : m_min(0), m_max(0) {
		}

		Normalizer(const Vector& min, const Vector& max) : m_min(min), m_max(max) {
		}

		void save(std::string filename) const
		{
			std::ofstream ofs(filename.c_str());
			boost::archive::text_oarchive oa(ofs);
			oa << const_cast<const Normalizer&>(*this);
		}

		template<class Archive> void serialize(Archive & ar, const unsigned int version) {
			ar & m_min;
			ar & m_max;
		}

		void calculate_from_input(const Matrix& input) {
			using namespace boost::numeric::ublas;

			m_min = Vector(input.size2());
			m_max = Vector(input.size2());

			for (unsigned int i=0; i<input.size2(); i++) {
				matrix_column<const Matrix> mc(input, i);
				m_min(i) = *std::min_element(mc.begin(), mc.end());
				m_max(i) = *std::max_element(mc.begin(), mc.end());
			}
		}


		Vector deNormalize(const Vector& input) const {
			using namespace boost::numeric::ublas;
			return element_prod(input,m_max - m_min) + m_min;
			
		}
		
		Vector normalize(const Vector& input) const {
			using namespace boost::numeric::ublas;
			return element_div(input - m_min , m_max - m_min);
		}
		
		Matrix normalize(const Matrix& input) const {
			
			Matrix ret(input.size1(), input.size2());
			for (unsigned int i=0; i<input.size1(); i++)
				row(ret,i) = normalize(row(input,i));
			
			return ret;
		}
		
		Matrix deNormalize(const Matrix& input) const {
			
			Matrix ret(input.size1(), input.size2());
			for (unsigned int i=0; i<input.size1(); i++)
				row(ret,i) = deNormalize(row(input,i));
			
			return ret;
		}
		
		~Normalizer() {};
	protected:
		Vector m_min; 
		Vector m_max;
};

#endif
