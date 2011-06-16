#include "rbfn_bindings.h"
#include "normalizer.h"
#include <boost/python.hpp>
#include <cstdlib>
#include <ctime>

namespace bp = boost::python;

BOOST_PYTHON_MODULE(libpyrbfnet) {

	srand(time(NULL));
    import_array();

    typedef RBFN_Wrapper::vt vt;
    bp::class_<RBFN_Wrapper> ("PyRBFNetwork", bp::init< unsigned int, unsigned int, vt >())
		.def(bp::init<std::string>())
		.def("init_weights", &RBFN_Wrapper::init_weights)
		.def("output", &RBFN_Wrapper::output)
		.def("lsqtrain", &RBFN_Wrapper::lsqtrain)
		.def("select_random_kernels", &RBFN_Wrapper::select_random_kernels)
		.def("get_weights", &RBFN_Wrapper::weights)
		.def("get_kernels", &RBFN_Wrapper::kernels)
		.def("set_weights", &RBFN_Wrapper::set_weights)
		.def("set_kernels", &RBFN_Wrapper::set_kernels)
		.def("get_num_kernels", &RBFN_Wrapper::num_kernels)
		.def("get_input_size", &RBFN_Wrapper::input_size)
		.def("get_output_size", &RBFN_Wrapper::output_size)
		.def("get_sigma", &RBFN_Wrapper::sigma)
		.def("set_sigma", &RBFN_Wrapper::setSigma)
		.def("output_conf", &RBFN_Wrapper::output_conf)

		.def_pickle(rbfnetwork_pickle_suite())
		;

    bp::class_<Normalizer_Wrapper>("PyNormalizer")
		.def(bp::init<PyObject*, PyObject*>())
		.def("calculate_from_input", &Normalizer_Wrapper::calculate_from_input)
		.def("denormalize", &Normalizer_Wrapper::denormalize)
		.def("normalize", &Normalizer_Wrapper::normalize)
		.def("min", &Normalizer_Wrapper::min)
		.def("max", &Normalizer_Wrapper::max)

		.def_pickle(normalizer_pickle_suite())
		;

}
