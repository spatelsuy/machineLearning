# machineLearning
For a traditional programmer, it is a puzzle how machine learning models actually learn. At the end, it is a program written in a programming language. So, how does the program behind a machine learning model differ from traditional C/C++/Fortran/Basic/Pascal/COBOL programs?  

The simple answer I can think of is: Machine learning is nothing more than a clever way to make a program adjust its behavior based on data patterns, instead of hardcoding every possible rule. For example, you are not telling the machine the rules of addition, instead you are letting the program figure out the rules from the pattern or examples you provided.  

Let's assume the ML program is written in Python. Now, the obvious question is, how does the Python program figure out the rules? While the developer is using various logical statements, the core resides in mathematics. There are innumerable mathematical formulas behind it.  

## How Machine Learning "Learns" Behind the Scenes
Here's one common approach:

- Input Data: It starts with examples (e.g., [1, 2] → 3).  
- Initialize Weights: The model begins with random weights and bias.    
- Make Predictions: Using a formula like Y = W1·x1 + W2·x2 + b (a linear formula).  
- Compare with Actual Output: Calculate the difference (error).  
- Adjust Weights: Using methods like Gradient Descent to minimize the error or use normal equation.  

**Can we use a linear formula for all?**  
No, we cannot. For example, a^b is not a linear graph. For a^b, we can use Symbolic models. However, we still calculate the error (Mean Squared Error (MSE)).

For an old-school programmer, here is a Python Code to get an understanding of machine learning as a starting point. In the code, I am using linear regression, exponential regression, power regression, symbolic model and using normal equation instead of gradient descent. My statistics friends can relate these statistical models. 

While the code developed in Python can be written in C/C++, Python already has libraries to simplify your coding, instead of you writing code to transpose a matrix or perform dot products.
