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

// #define CONSTANT


RbfNetwork::RbfNetwork(unsigned int input_size, unsigned int output_size, vt learning_rate, vt sigma) :
	m_learning_rate(output_size), 
#ifdef CONSTANT
	m_weights(ublas::zero_matrix<Matrix::value_type>(1,output_size)),
	m_oldw(ublas::zero_matrix<Matrix::value_type>(1,output_size))
#else
	m_weights(ublas::zero_matrix<Matrix::value_type>(0,output_size)),
	m_oldw(ublas::zero_matrix<Matrix::value_type>(0,output_size))
#endif
	
{
	__static__generator.seed(static_cast<unsigned int>(std::time(0)));
	m_input_size = input_size;
	m_output_size = output_size;
	m_nimin = learning_rate;
	m_sigma = sigma*sigma;
	m_lveta = 0;
	m_addkernels = true;
	m_firsttrain = true;
	for (unsigned int i=0; i<output_size; i++) {
		m_learning_rate(i) = 0.1;
	}
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
	row(m_oldw,weightnumber) = ublas::zero_vector<Vector::value_type>(m_oldw.size2());
	row(m_oldDerr,weightnumber) = ublas::zero_vector<Vector::value_type>(m_oldDerr.size2());
	
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

void RbfNetwork::add_kernel(const Vector& center)
{
	if (m_addkernels) {
		vt dist = minDistFromKernel(center);
		if (dist > 2.0*m_sigma) {
			m_kernels.resize(m_kernels.size1()+1,m_input_size,true);	
			row(m_kernels,m_kernels.size1()-1) = center;
			
			m_weights.resize(m_weights.size1() + 1,m_output_size,true);
			m_oldw.resize(m_weights.size1(),m_weights.size2(),true);
			m_oldDerr.resize(m_weights.size1(),m_weights.size2(),true);
			
			init_weights(0,0,m_weights.size1()-1);
		}
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
// 	m_weights.resize(1+size,m_output_size,false);
// 	m_oldw.resize(m_weights.size1(),m_weights.size2(),true);
// 	m_oldDerr.resize(m_weights.size1(),m_weights.size2(),true);
	
// 	m_kernels.assign(ublas::zero_matrix<Matrix::value_type>(size,m_input_size));
#ifdef CONSTANT
	m_weights = ublas::zero_matrix<Matrix::value_type>(1+size,m_output_size);
#else
	m_weights = ublas::zero_matrix<Matrix::value_type>(size,m_output_size);
#endif
	m_oldw = ublas::zero_matrix<Matrix::value_type>(m_weights.size1(),m_weights.size2());
	m_oldDerr = ublas::zero_matrix<Matrix::value_type>(m_weights.size1(),m_weights.size2());
		
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

void RbfNetwork::remove_kernels(const std::set<unsigned int >& index)
{
	unsigned int to_swap = m_kernels.size1() - 1;
	for (std::set<unsigned int>::const_iterator i = index.begin(); i != index.end(); i++) {
		//swapping the element to delete with the last one
		
#ifdef CONSTANT		
		if (*i==0) //that's the bias
			continue;

		//the weights have always one element more than the kernels for the bias, which is never removed
		ublas::row(m_weights,*i + 1).swap(ublas::row(m_weights,to_swap + 1));
#else
		ublas::row(m_weights,*i).swap(ublas::row(m_weights,to_swap));
		ublas::row(m_kernels,*i ).swap(ublas::row(m_kernels,to_swap));
#endif
		
		to_swap--;
	}
	m_weights.resize(m_weights.size1() - index.size(), m_weights.size2(), true);
	m_kernels.resize(m_kernels.size1() - index.size(), m_kernels.size2(), true);
}

RbfNetwork::Vector RbfNetwork::first_layer_output(const Vector& input) const
{
#ifdef CONSTANT
	Vector res(1 + m_kernels.size1());
	res(0) = 1.0;
#else
	Vector res(m_kernels.size1());
#endif
	
	for (unsigned int i=0; i<m_kernels.size1(); i++) {
		Vector tmp(input - row(m_kernels,i));
		vt d = sum(element_prod(tmp,tmp));
#ifdef CONSTANT
		res(i+1) = exp(-d/m_sigma);
#else
		res(i) = exp(-d/m_sigma);
#endif
	}
	return res;
}

RbfNetwork::Matrix RbfNetwork::first_layer_output(const Matrix& inputs) const 
{
#ifdef CONSTANT	
	Matrix res(inputs.size1(),1 + m_kernels.size1()); //including bias
#else
	Matrix res(inputs.size1(),m_kernels.size1());
#endif


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
#ifdef CONSTANT		
		//see output_conf for a single value to understand what's down here		
		conf(i) = norm_inf(subrange(ublas::row(firsto,i),1,firsto.size2()));  //taking the bias out
#else
		conf(i) = norm_inf(ublas::row(firsto,i));
#endif		
	}
	
	return boost::make_tuple(out,conf);
}

boost::tuples::tuple<RbfNetwork::Vector, RbfNetwork::vt, unsigned int> RbfNetwork::output_conf_index(const Vector& input) const
{
	Vector firsto = first_layer_output(input);
	Vector out = prod(firsto,m_weights);	
	
#ifdef CONSTANT	
	vt conf = norm_inf(subrange(firsto,1,firsto.size()));  //taking the bias out
	unsigned int index = index_norm_inf(subrange(firsto,1,firsto.size()));
#else
	vt conf = norm_inf(firsto);
	unsigned int index = index_norm_inf(firsto);
#endif
	
	return boost::make_tuple(out,conf, index);
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

RbfNetwork::Vector RbfNetwork::gdtrain(const Vector& input, const Vector& target)
{


	//do we add a kernel?
	if (m_addkernels) {
		vt dist = minDistFromKernel(input);
		if (dist > 2.0*m_sigma) {
			add_kernel(input);
		}
	}
	
	//calculating the network oputput
	Vector kernels_outputs = first_layer_output(input);
	Matrix Derr(m_weights.size1(),m_weights.size2());
	Vector netout = prod(kernels_outputs,m_weights);
	
	//training the second layer
	for (unsigned int i=0; i<m_output_size; i++) {		
		column(Derr,i) = -(target(i) - netout(i))*kernels_outputs;
		if (m_firsttrain) {
			m_learning_rate(i) = armijo(column(Derr,i),Vector(),column(m_weights,i),column(m_oldw,i),kernels_outputs,target(i));
			m_firsttrain = false;
		}
		else
			m_learning_rate(i) = armijo(column(Derr,i),column(m_oldDerr,i),column(m_weights,i),column(m_oldw,i),kernels_outputs,target(i));
		
		column(m_oldw,i) = column(m_weights,i);
		column(m_weights,i) -= m_learning_rate(i)*column(Derr,i);
	}
	
	m_oldDerr = Derr;
	//calculating the error
	Vector dist(target - prod(kernels_outputs,m_weights));
	
	return dist;
}

RbfNetwork::vt RbfNetwork::armijo(const Vector& Derr,
		const Vector& oldDerr,
		const Vector& w,
		const Vector& oldw,
		const Vector& kernels_outputs,
		RbfNetwork::vt target)
{
	
	vt ni;
	vt num =0, den = 0;
	if (! oldDerr.empty() ) {
		num = norm_2(w - oldw);
		den = norm_2(Derr - oldDerr);
		if (den == 0)
			ni = 0;
		else
			ni =  num / den;
	}
	else
		ni = m_nimin;
	
	if (isnan(ni)) {
		ni=0;
	}
	if (ni == 0)
		ni = m_nimin;
	else {
		while (ni < m_nimin)
			ni = 2.0*ni;
	}
		
	vt err = target - inner_prod(kernels_outputs,w);
	err = 0.5*err*err;
		
	vt errni = target - inner_prod(kernels_outputs,w-(ni*Derr));
	errni = 0.5*errni*errni;
	
	vt th = -0.5 * ni * inner_prod(Derr,Derr);
	while ( (errni - err) > th) {
		if (errni == err)
			break;
		ni = ni/2.0;
		errni = target - inner_prod(kernels_outputs,w -(ni*Derr));
		errni = 0.5*errni*errni;
		th = -0.5 * ni * inner_prod(Derr,Derr);
	}
		
	return ni;	
}

RbfNetwork::Vector RbfNetwork::gdtrain(const Matrix& input, const Matrix& target)
{
	Vector ris(m_output_size);
	for (unsigned int i=0; i<input.size1()-1; i++) {
		std::cout<<"Iteration "<<i<<"\n";
		gdtrain(row(input,i),row(target,i));
	}
	return gdtrain(row(input,input.size1()-1),row(target,input.size1()-1));
}


/*!
    \fn RbfNetwork::RbfNetwork::minDistFromKernel(const Vector& vec) const
 */
RbfNetwork::vt RbfNetwork::minDistFromKernel(const Vector& vec) const
{
	vt dist = 1e20;
	for (unsigned int i=0; i<m_kernels.size1(); i++) {
		vt n = norm_2(vec - row(m_kernels,i)); //norm_1 is more efficient that norm_2
		if (n < dist) {
			dist = n;
		}
	}
	
	return dist;
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

unsigned int RbfNetwork::remove_unused_nodes(Matrix input, vt thr)
{
	//I admit this is one of the worst function I ever wrote
	std::list<unsigned int> toremove;
	std::set<unsigned int> remove_set;
	for (unsigned int i=0; i<m_kernels.size1(); i++)
		toremove.push_front(i);
	
	vt totalreduced = 0;
	for (std::list<unsigned int>::iterator node_i = toremove.begin(); node_i != toremove.end(); node_i++) {
		if (totalreduced >= thr)
			break;
		bool removeit = true;
		vt value = 0;
		for (unsigned int i=0; i<input.size1(); i++) {
			//finding the kernel value
			Vector tmp(ublas::row(input,i) - row(m_kernels,*node_i));
			value = ublas::norm_inf(exp(-sum(element_prod(tmp,tmp))/m_sigma) * ublas::row(m_weights,*node_i+1) );
			if ( (value + totalreduced) > thr) {
				removeit = false;
				break;
			}
		}
		if (removeit) {
			totalreduced += value;
  			remove_set.insert(*node_i);
		}
	}
	
	remove_kernels(remove_set);
	return remove_set.size();
}
