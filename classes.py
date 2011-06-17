import libpyrbfnet
import numpy

class DimensionError(Exception):
    pass
class IndexError(Exception):
    pass

class RbfNetwork(libpyrbfnet.PyRBFNetwork):
    """
    A multi-output radial basis function network
    """
    def output(self,  input):
        """
        Calculates the output of the network given input

        input has to be a 1d or 2d ndvector
        If input is a matrix, every row should be a  multi-dimensional variable
        Returns an ndarray of the same size of input
        """
        newinput = numpy.asarray(input)
        if newinput.ndim == 1: #dealing with a vector
            newinput = newinput.reshape( (1, newinput.shape[0]))
        elif newinput.ndim == 2: #matrix
            pass
        else:
            raise DimensionError("input has to be either a vector or a matrix")
        if newinput.shape[1] != self.input_size:
            raise DimensionError("input dimension differs from the RBF one")

        return libpyrbfnet.PyRBFNetwork.output(self, newinput)

    def output_conf(self, input):
        newinput = numpy.asarray(input)
        if newinput.ndim == 1: #dealing with a vector
            newinput = newinput.reshape( (1, newinput.shape[0]))
        elif newinput.ndim == 2: #matrix
            pass
        else:
            raise DimensionError("input has to be either a vector or a matrix")
        if newinput.shape[1] != self.input_size:
            raise DimensionError("input dimension differs from the RBF one")
        
        return libpyrbfnet.PyRBFNetwork.output_conf(self, newinput)

    def __call__(self, input):
        return self.output(input)

    def lsqtrain(self,  input,  output):
        """
        Perform least sqare training over input/outputs

        input/output has to be a 2d ndvector, and every row should be a  multi-dimensional variable
        Returns an ndarray of the same size of input/output
        """
        newinput = numpy.asarray(input)
        newoutput = numpy.asarray(output)

        if newinput.ndim != newoutput.ndim:
            raise DimensionError("input and output must have the same shape")

        if newinput.ndim != 2:
            raise DimensionError("input has to be a matrix")

        if newinput.shape[1] != self.input_size:
            raise DimensionError("input dimension differs from the RBF one")
        if newoutput.shape[1] != self.output_size:
            raise DimensionError("output dimension differs from the RBF one")

        if newoutput.shape[0] != newinput.shape[0]:
            raise DimensionError("input and output must have the same number of rows ")

        return libpyrbfnet.PyRBFNetwork.lsqtrain(self, newinput, newoutput)

    def select_random_kernels(self,  input,  number):
        """
        Select number vectors from input to be used as kernels
        """
        newinput = numpy.asarray(input)
        if newinput.ndim != 2:
            raise DimensionError("input has to be a matrix")

        if number > input.shape[0]:
            raise IndexError("asking for more elements that in input")

        if newinput.shape[1] != self.input_size:
            raise DimensionError("input dimension differs from the RBF one")
        return libpyrbfnet.PyRBFNetwork.select_random_kernels(self, newinput,  number)

    weights = property(libpyrbfnet.PyRBFNetwork.get_weights, libpyrbfnet.PyRBFNetwork.set_weights)
    kernels = property(libpyrbfnet.PyRBFNetwork.get_kernels, libpyrbfnet.PyRBFNetwork.set_kernels)
    num_kernels = property(libpyrbfnet.PyRBFNetwork.get_num_kernels)
    input_size = property(libpyrbfnet.PyRBFNetwork.get_input_size)
    output_size = property(libpyrbfnet.PyRBFNetwork.get_output_size)
    sigma = property(libpyrbfnet.PyRBFNetwork.get_sigma, libpyrbfnet.PyRBFNetwork.set_sigma)
    

def classes2matrix(classes):
    """
    Convert a vector of classes to a matrix suitable for
    RBF use.

    Each class is an integer number, including zero.
    The number N of classes is taken as max(classes) + 1.
    The output is a MxN matrix, where M is len(classes)
    """

    classes = numpy.asarray(classes,  dtype=numpy.int).flatten()
    M = len(classes)
    N = numpy.max(classes) + 1
    out = numpy.zeros( (M,  N),  dtype=numpy.float32)
    out[ numpy.arange(M),  classes] = 1
    return out

def matrix2classes(mat):
    return numpy.argmax(mat, 1).reshape((mat.shape[0], 1))

class RbfClassifier(RbfNetwork):

    def __init__(self,  *args):
        if len(args) == 1:
            RbfNetwork.__init__(self,  args[0])
        elif len(args) == 3:
            (num_input,  num_classes, sigma) = args
            if num_classes <= 1:
                raise DimensionError("num_classes must be > 1")
            RbfNetwork.__init__(self,  num_input,  num_classes, sigma)
            self.__indeces = []
        else:
            raise TypeError("Wrong number of arguments, it should be either 3 or 1")

    def select_random_kernels(self,  input,  number):
		newinput = numpy.asarray(input)
		if newinput.ndim != 2:
			raise DimensionError("input has to be a matrix")
		if newinput.shape[1] != self.input_size:
			raise DimensionError("input dimension differs from the RBF one")
		max_elements = newinput.shape[0]
		indeces = set()
		while len(indeces) != number:
			indeces.add(numpy.random.randint(0, max_elements))

		self.__indeces = list(indeces)
		self.kernels = newinput[self.__indeces,  :]

    def lsqtrain(self,  input,  output):

        newinput = numpy.asarray(input)
        newoutput = classes2matrix(output)

        if newinput.ndim != newoutput.ndim:
            raise DimensionError("input and output must have the same shape")

        if newinput.ndim != 2:
            raise DimensionError("input has to be a matrix")

        if newinput.shape[1] != self.input_size:
            raise DimensionError("input dimension differs from the RBF one")
        if newoutput.shape[1] != self.output_size:
            raise DimensionError("output dimension differs from the RBF one")

        if newoutput.shape[0] != newinput.shape[0]:
            raise DimensionError(
                "input and output must have the same number of rows ")

        if newoutput.shape[1] <=0:
            raise DimensionError("output has <=0 columns")
        if newinput.shape[1] <=0:
            raise DimensionError("input has <=0 columns")

        newweights = numpy.vstack( (numpy.zeros((1,newoutput.shape[1])), newoutput[self.__indeces,  :]) )
        self.weights = newweights

        netout = self.output(newinput)
        return netout != self.matrix2classes(newoutput)

    def output(self,  input):
        output = RbfNetwork.output(self,  input)
        return self.matrix2classes(output)

    def raw_output(self,  input):
        return RbfNetwork.output(self,  input)

    def __call__(self, input):
        return self.output(input)

    def output_conf(self, input):
        output, conf = RbfNetwork.output_conf(self,  input)
        return self.matrix2classes(output), conf

    @staticmethod
    def matrix2classes(m):
        return matrix2classes(m)

    @staticmethod
    def classes2matrix(c):
        return classes2matrix