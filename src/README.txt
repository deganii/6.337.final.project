------------------
Location of BFGS Implementation:
------------------

./optimizers/ext_bfgs.py

------------------
Location of Gradient Descent Implementation:
------------------

./optimizers/ext_grad.py

------------------
Usage: Toy Problem
------------------

To run the BFGS optimizer on a small example problem, type:

python simple_optim_test.py

This will solve the Ax = b equation:

| 1  2  3|   | x1 |     | 9 |
| 2 -1  0| * | x2 |  =  | 5 |
| 3  0  1|   | x3 |     | 3 |

The correct solution is [x1, x2, x3] = [2, -1, 3]

------------------
Usage: MNIST
------------------

To run the BFGS optimizer on a small example problem, type:

python logistic_mnist.py

This will run BFGS optimizer on the MNIST dataset

------------------
Run other algorithms
------------------

To change the algorithm that is ran to Adam optimizer or Gradient descent, edit

logistic_mnist.py adam
logistic_mnist.py ext_grad

Similarly, run the following on the toy problem:

python simple_optim_test.py adam
python simple_optim_test.py ext_grad

