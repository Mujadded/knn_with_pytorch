import torch

class K_NN:

    def __init__(self, x_train: torch.Tensor, y_train: torch.Tensor):
        """
        Creates a K-Nearest Neighbor Instance and save X Train and Y Train in memory
        """
        self.x_train = x_train
        self.y_train = y_train
    
    def predict(self, x_test: torch.Tensor, k: int = 1):
        """
        Calculate the distance of all test and training 
        Then predict the label depending on the K value
        """

        # The number of training and test for use
        num_train = self.x_train.shape[0]
        num_test = x_test.shape[0]

        # We will calculate L2 Distance which means it will be
        # sum(sqrt((X_Train - X_Test)**2))
        # As we are trying to vectorize it so that no loop is used
        # We can break the equation into:
        # sum(X_Train**2) - 2*X_Train*X_Test + sum(X_Test**2)
        
        # Getting X_Train's Flat verison
        flat_x_train = self.x_train.view(num_train,-1)
        
        # Getting X_Test flat version
        flat_x_test = x_test.view(num_test,-1)

        # Calculating sum(X_Train**2)
        # Here we need to reshape it first because we want to vectorizedly computer
        # an array where for every X_Train[i] there is a row of difference with test
        # Althought we have made the equation expanded but that doesnt change what
        # Needed to be done
        flat_x_train_sqared= (flat_x_train.view(num_train,1,-1)**2).sum(dim=2)
        
        # Calcualting sum(X_Test**2)
        flat_x_test_sqared= (flat_x_test**2).sum(dim=1)

        # Calculating 2*X_Train*X_Test
        multi=2*flat_x_train.matmul(flat_x_test.t())

        # so getting everything together
        dists=(flat_x_train_sqared-multi+flat_x_test_sqared).sqrt()


        # Now we are ready to Predict

        # Setting O as default prediction (empty values)
        y_pred = torch.zeros(num_test, dtype=torch.int64)

        # Getting the k number of lowest values per row in column
        _, indices = dists.topk(k,largest=False,dim=0)
        for i in range(num_test):

            pred_index = torch.argmax(torch.bincount(self.y_train[indices[:,i]])) 
            y_pred[i] = pred_index
        
        return y_pred