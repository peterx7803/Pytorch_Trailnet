import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.backends.cudnn as cudnn

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 32, 4)
        self.conv2 = nn.Conv2d(32, 32, 4)
	self.conv3 = nn.Conv2d(32, 32, 4)
	self.conv4 = nn.Conv2d(32, 32, 4)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(288, 200)
        self.fc2 = nn.Linear(200, 7)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
	#print x 
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
	#print x
	x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
	#print x
	x = F.max_pool2d(F.relu(self.conv4(x)), (2, 2))
	#print x

        x = x.view(-1, self.num_flat_features(x))
	#print x
        x = F.relu(self.fc1(x))
	#print x
        x = F.relu(self.fc2(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

cudnn.benchmark = True

net = Net()
net.cuda()
print(net)


params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight

tmp1,tmp2 = 0,0
test_step = 10
batchsz = 1
Drytest = 5
for i in range(1,Drytest + test_step+1):
	print 'i=', i
	# Forward
	input = Variable(torch.randn(batchsz, 1, 101, 101)).cuda()   ##Dummy input
	torch.cuda.synchronize()
	tf = time.time()
	out = net(input)
	elapsed1= time.time() -tf
	torch.cuda.synchronize()
	tmp1 += elapsed1
	#print(out)
	
	# Backward
	'''
	net.zero_grad()
	out.backward(torch.randn(1, 7))			
	'''
	# Loss function
	output = net(input)
	target = Variable(torch.randn(batchsz, 7)).cuda()  # a dummy target, for example

	Optimizer = torch.optim.SGD(net.parameters(), lr = 0.001)
	criterion = nn.MSELoss()
	loss = criterion(output, target)
	print(loss)

	#print(loss.grad_fn)  # MSELoss
	#print(loss.grad_fn.next_functions[0][0])  # Linear
	#print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

	# Back prop.

	net.zero_grad()     # zeroes the gradient buffers of all parameters
	print('conv1.bias.grad before backward')
	print(net.conv1.bias.grad)
	torch.cuda.synchronize()
	tb = time.time()
	loss.backward()
	elapsed2 = time.time()- tb
	torch.cuda.synchronize()
	tmp2 += elapsed2
	
	#print('conv1.bias.grad after backward')
	#print(net.conv1.bias.grad)
	
	if i<= Drytest:
		tmp1,tmp2 = 0, 0
	print tmp1, ' ', tmp2

print 'fp:' ,tmp1/test_step, ' bp: ',tmp2/test_step





