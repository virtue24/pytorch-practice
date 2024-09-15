import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

weight = 0.7
bias = 0.3

start,end,step = 0,1,0.02

X = torch.arange(start=start, end=end, step=step).unsqueeze(1)
Y = weight * X + bias

train_split = 0.8
X_train = X[:int(train_split * len(X))]
Y_train = Y[:int(train_split * len(Y))]
X_test = X[int(train_split * len(X)):]
Y_test = Y[int(train_split * len(Y)):]

def plot_test_and_train_data(test_x, test_y, train_x, train_y):
    import matplotlib.pyplot as plt
    plt.scatter(train_x, train_y, label="Train data")
    plt.scatter(test_x, test_y, label="Test data")
    plt.legend()
    plt.show()
#plot_test_and_train_data(X_test, Y_test, X_train, Y_train)

class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.rand(1, requires_grad=True, dtype=torch.float))
        self.bias = torch.nn.Parameter(torch.rand(1, requires_grad=True, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x + self.bias

torch.manual_seed(42)
model = LinearRegressionModel()
print(list(model.parameters()))

with torch.inference_mode():
    Y_pred = model(X_test)

def plot_predictions(test_x, test_y, predicted_y):
    import matplotlib.pyplot as plt
    plt.scatter(test_x, test_y, label="Actual")
    plt.scatter(test_x, predicted_y, label="Predicted")
    plt.legend()
    plt.show()
#plot_predictions(X_test, Y_test, Y_pred)

print(model.state_dict())


loss_fn = torch.nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

epochs = 5000
for epoch in range(epochs):
    model.train() # set the model to training mode (gradient computation is enabled)

    # 1. Forward pass: Compute predicted y by passing x to the model
    Y_pred = model(X_train)
   
    # 2. Compute and print loss
    loss = loss_fn(Y_pred, Y_train)
    print('Epoch:', epoch, 'Loss:', loss.item())

    # 3. Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()

    # 4. Compute the gradient of the loss with respect to all the learnable parameters
    loss.backward()

    # 5. Update the weights
    optimizer.step()




