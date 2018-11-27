
class EarlyStopper(object):
    def __init__(self, criteria, patience, high_is_better=True):
        """
        Object used to monitor a criteria and trigger early stopping
        Arguments:
            criteria (str):          Criteria that will monitored through training
            patience (int):          Delay before stopping training if not improving
            higher_is_better (bool): Wheter a high value of the criteria is better
        """
        self.criteria = criteria
        self.high_is_better = high_is_better
        self.patience = patience

        self.patience_counter = self.patience
        self.new_best = False
        self.stop = False
        self.best_id = 0
        self.best_value = -float('inf') if high_is_better else float('inf')

    def update(self, stats, epoch_id, batch_id):
        if stats[self.criteria] > self.best_value:
            self.best_epoch_id = epoch_id
            self.best_batch_id = batch_id
            self.best_value = stats[self.criteria]
            self.patience_counter = self.patience
            self.new_best = True
        else:
            self.new_best = False
            self.patience_counter -= 1
            self.stop = True if self.patience_counter <=0 else False

    def print_stop(self):
        print('Stopping training due to lack of improvement..')
        print('Best performing model:\t epoch {}, batch {} ({} {:7.4f})'.format(
            early_stopper.best_epoch_id,
            early_stopper.best_batch_id,
            early_stopper.criteria,
            early_stopper.best_value))
