class Validation: 
    def __init__(self, predictions):
        self.predictions = predictions

    def val_negative(self):
        if (self.predictions['forecasted_tickets'].values < 0).any():
            return '[X] WARNING! The predicted numbers of tickets have negative values!'
        else: 
            return '[OK] NEGATIVE PREDICTION VALIDATION PASSED'


