//
// C++ Implementation: rbfmonetwork
//
// Description: 
//
//
// Author:  <>, (C) 2009
//
// Copyright: See COPYING file that comes with this distribution
//
//
#include <rbfnetwork.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <list>
#include <cmath>

#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/random.hpp>
#include <boost/numeric/bindings/lapack/gelss.hpp>
#include <boost/numeric/bindings/traits/ublas_matrix.hpp>
#include <boost/numeric/bindings/traits/matrix_traits.hpp>
#include <boost/numeric/bindings/lapack/workspace.hpp>


namespace ublas = boost::numeric::ublas;
namespace lapack = boost::numeric::bindings::lapack;

typedef boost::minstd_rand base_generator_type;
static base_generator_type __static__generator(static_cast<unsigned int>(std::time(0)));


RbfNetwork::RbfNetwork(unsigned int input_size, unsigned int output_size, vt sigma) :
	m_weights(ublas::zero_matrix<Matrix::value_type>(1,output_size))
{
	__static__generator.seed(static_cast<unsigned int>(std::time(0)));
	m_input_size = input_size;
	m_output_size = output_size;
	m_sigma = sigma*sigma;
}

RbfNetwork::RbfNetwork(std::string filename) {
	
// 	__static__generator.seed(static_cast<unsigned int>(std::time(0)));
	std::ifstream ifile(filename.c_str());
	assert(ifile.is_open());
	boost::archive::text_iarchive ia(ifile);
	ia >> *this;	
}

RbfNetwork::~RbfNetwork()
{
}

void RbfNetwork::init_weights(vt a, vt b, unsigned int weightnumber)
{
	if (weightnumber > m_weights.size1()) {
		throw rbfn_value_exception("Wrong index!");
	}
	if (a == b) {
		row(m_weights,weightnumber) = ublas::zero_vector<Vector::value_type>(m_weights.size2());
		return;
	}
	boost::uniform_real<> uni_dist(a,b);
	boost::variate_generator<base_generator_type&, boost::uniform_real<> > uni(__static__generator, uni_dist);
	
	for (unsigned int i=0; i<m_weights.size2(); i++) {
		m_weights(weightnumber,i) = (b-a)*uni() + a;
		
	}
}

void RbfNetwork::select_random_kernels(const Matrix& input, unsigned int size)
{
	if (input.size2() != m_input_size) {
		std::stringstream msg;
		msg<<"Wrong size of input matrix. Expected "<<m_input_size<<", received "<< input.size2();
		throw rbfn_value_exception(msg.str());
	}
	m_kernels.resize(size,m_input_size,false); //they will be overwritten very soon
	m_weights = ublas::zero_matrix<Matrix::value_type>(1+size,m_output_size);
		
	boost::uniform_smallint<> uni_dist(0,input.size1() - 1);
	boost::variate_generator<base_generator_type&, boost::uniform_smallint<> > rand(__static__generator, uni_dist);
	
	std::set<int> draws;
	while (draws.size() != size ) {
		int n = rand();
		draws.insert(n);
	}
	
	unsigned int pos = 0; 
	for (std::set<int>::iterator i = draws.begin(); i != draws.end(); i++) {
		ublas::row(m_kernels,pos) = ublas::row(input,*i);
		pos++;
	}
}

RbfNetwork::Vector RbfNetwork::first_layer_output(const Vector& input) const
{
	Vector res(1 + m_kernels.size1());
	res(0) = 1.0;
	
	for (unsigned int i=0; i<m_kernels.size1(); i++) {
		Vector tmp(input - row(m_kernels,i));
		vt d = sum(element_prod(tmp,tmp));
		res(i+1) = exp(-d/m_sigma);
	}
	return res;
}

RbfNetwork::Matrix RbfNetwork::first_layer_output(const Matrix& inputs) const 
{
	Matrix res(inputs.size1(),1 + m_kernels.size1()); //including bias

	for (unsigned int i=0; i<inputs.size1(); i++)
		ublas::noalias(row(res,i)) = first_layer_output(row(inputs,i));
	return res;
}

RbfNetwork::Vector RbfNetwork::output(const Vector& input) const
{
	if (input.size() != m_input_size) {
		std::stringstream msg;
		msg<<"Wrong size of input. Expected "<<m_input_size<<", received "<< input.size();
		throw rbfn_value_exception(msg.str());
	}
	return prod(first_layer_output(input),m_weights);
}

boost::tuples::tuple<RbfNetwork::Vector, RbfNetwork::vt> RbfNetwork::output_conf(const Vector& input) const
{
	Vector firsto = first_layer_output(input);
	Vector out = prod(firsto,m_weights);
	
	vt conf = norm_inf(subrange(firsto,1,firsto.size()));  //taking the bias out
	
	return boost::make_tuple(out,conf);
}

boost::tuples::tuple<RbfNetwork::Matrix, RbfNetwork::Vector> RbfNetwork::output_conf(const Matrix& input) const
{
	Matrix firsto = first_layer_output(input);
	Matrix out = prod(firsto,m_weights);
	
	Vector conf(firsto.size1());
	for (unsigned int i=0; i<firsto.size1(); i++) {
		//see output_conf for a single value to understand what's down here		
		conf(i) = norm_inf(subrange(ublas::row(firsto,i),1,firsto.size2()));  //taking the bias out
	}
	
	return boost::make_tuple(out,conf);
}

RbfNetwork::Matrix RbfNetwork::output(const Matrix& input) const
{
	return prod(first_layer_output(input),m_weights);
}

RbfNetwork::Matrix RbfNetwork::lsqtrain(const Matrix& input,const Matrix& target)
{
	if (input.size2() != m_input_size) {
		std::stringstream msg;
		msg<<"Wrong size of input matrix. Expected "<<m_input_size<<", received "<< input.size2();
		throw rbfn_value_exception(msg.str());
	}
	if (input.size1() != target.size1()) {
		std::stringstream msg;
		msg<<"Input matrix number of elements is "<<input.size1()
				<<", while the target matrix is "<< target.size1();
		throw rbfn_value_exception(msg.str());
	}
	if (target.size2() != m_output_size) {
		std::stringstream msg;
		msg<<"Wrong size of target matrix. Expected "<<m_output_size<<", received "<< target.size2();
		throw rbfn_value_exception(msg.str());
	}


// 	Vector res(m_weights.size2());
	ublas::matrix<Matrix::value_type,ublas::column_major> A(first_layer_output(input));
	ublas::matrix<Matrix::value_type,ublas::column_major> b(target);
	
	lapack::optimal_workspace w;
	lapack::gelss(A,b,w);
	
	m_weights = ublas::subrange(b,0,A.size2(),0,b.size2());
	
	return target - prod(first_layer_output(input),m_weights);

}

unsigned int RbfNetwork::num_kernels() const
{
	return m_kernels.size1();
}


void RbfNetwork::save(std::string filename) const
{
	std::ofstream ofs(filename.c_str());
	boost::archive::text_oarchive oa(ofs);
	oa << const_cast<const RbfNetwork&>(*this);
}
