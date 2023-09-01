"""
The implementation of Logistic Regression using JAX.
Optimization algorithm: Basic Gradient Descent.
"""

# import jax
import jax.numpy as jnp
from jax import grad, jit
from jax.scipy.special import expit
from sklearn.preprocessing import StandardScaler

from sklearn import datasets
from sklearn.model_selection import train_test_split

class LogisticRegression:
    def __init__(self, learning_rate=0.1, num_epochs=1000, penalty=None, alpha=0.1):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.penalty = penalty  # 'l2', 'l1', 'elasticnet', or None
        self.alpha = alpha  # Regularization strength (only used for L1 and Elastic Net)

    def logistic_regression(self, params, x):
        z = jnp.dot(x, params)
        return expit(z)

    
    def cross_entropy_loss(self, params, x, y):
        y_pred = self.logistic_regression(params, x)
        loss = -jnp.mean(y * jnp.log(y_pred) + (1 - y) * jnp.log(1 - y_pred))
        
        # Add L2 regularization penalty
        if self.penalty == 'l2':
            l2_penalty = 0.5 * jnp.sum(params[:-1] ** 2)  # Exclude bias term
            loss += self.alpha * l2_penalty
        # Add L1 regularization penalty
        elif self.penalty == 'l1':
            l1_penalty = jnp.sum(jnp.abs(params[:-1]))  # Exclude bias term
            loss += self.alpha * l1_penalty
        # Add Elastic Net regularization penalty
        elif self.penalty == 'elasticnet':
            l2_penalty = 0.5 * jnp.sum(params[:-1] ** 2)  # Exclude bias term
            l1_penalty = jnp.sum(jnp.abs(params[:-1]))  # Exclude bias term
            loss += self.alpha * (self.alpha * l2_penalty + (1 - self.alpha) * l1_penalty)

        return loss

    def fit(self, X_train, y_train):
        # Add a bias term (intercept) to the features
        X_train = jnp.hstack((X_train, jnp.ones((X_train.shape[0], 1))))

        # Initialize model parameters
        params = jnp.zeros(X_train.shape[1])

        # Define the gradient of the loss function
        grad_loss = grad(self.cross_entropy_loss)

        # Optimization loop
        for epoch in range(self.num_epochs):
            # Compute gradients
            grads = grad_loss(params, X_train, y_train)

            # Basic gradient descent update
            params -= self.learning_rate * grads    # Update parameters
            
            loss = self.cross_entropy_loss(params, X_train, y_train)
            # Compute and print the loss (optional)
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

        self.params = params

    def evaluate(self, X_test, y_test):
        # Add a bias term (intercept) to the features
        X_test = jnp.hstack((X_test, jnp.ones((X_test.shape[0], 1))))

        # Make predictions on the test set
        y_pred = self.logistic_regression(self.params, X_test)

        # Convert probabilities to binary predictions
        y_pred_binary = (y_pred >= 0.5).astype(int)

        # Calculate accuracy
        accuracy = jnp.mean(y_pred_binary == y_test)
        return accuracy
    

if __name__ == "__main__":
    # Load the Iris dataset
    data = datasets.load_iris()
    X, y = data.data, data.target

    # We are interested in classifying Iris-Virginica (class 2) vs. the rest
    y_binary = (y == 2).astype(int)
    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

    # Create and train the logistic regression model with regularization
    model = LogisticRegression()  # Adjust the alpha parameter for regularization
    model.fit(X_train, y_train)

    # Evaluate the model
    accuracy = model.evaluate(X_test, y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")