import numpy as np
import cPickle

execfile("__init__.py")

input1 = np.random.rand(20, 4)
output1 = np.random.rand(20, 1)
net1 = RbfNetwork(4, 1, 0, np.std(input1))
net1.select_random_kernels(input1, 2)
net1.lsqtrain(input1,  output1)
print "First output: "
print net1.output(input1)[1]

input2 = np.random.rand(20, 1)
output2 = np.random.rand(20, 1)
net2 = RbfNetwork(1,1, 0, np.std(input2))
net2.select_random_kernels(input2, 2)
net2.lsqtrain(input2,  output2)
print "Second output: "
print net2.output(input2)[1]

print "First net kernels: ", net1.kernels
s = cPickle.dumps(net1)
net3 = cPickle.loads(s)
print "Depickled net kernels: ", net3.kernels


classifier = RbfClassifier(2,2,0.1)
inputs = np.random.rand(1000,2)
outs = []
for i in inputs:
    if i[0] + i[1] > 0.5:
        outs.append(1)
    else:
        outs.append(0)

classifier.select_random_kernels(inputs, 3)
classifier.lsqtrain(inputs, outs)

trials = 1000
res = 0.0
for sgh in xrange(trials):
    vals = np.random.rand(2)
    outcome = np.sum(vals) > 0.5
    netout = classifier.output(vals)
    res += np.abs(netout - outcome)

print "Classification: ", 1.0 - (res / trials)

print "Done"
