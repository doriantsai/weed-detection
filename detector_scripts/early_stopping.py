# early stopping class from
# https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/

class EarlyStopping():
    """ early stopping class"""

    def __init__(self,
                patience=5,
                min_delta=0):
        """
        patience - how many epochs to wait for if not improving
        min_delta - min diff between loss and new loss to be considered for improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss >= self.min_delta:
            self.best_loss = val_loss
        elif self.best_loss - val_loss < self.min_delta
            self.counter += 1
            print('Early stopping counter {} of {}'.format(self.counter,
                                                           self.patience))
            if self.counter >= self.patience:
                print('Early stopping NOW')
                self.early_stop = True

# end class
