class Trainer(object):

    '''Class to train a model '''
    def __init__(self,model,dataset, **kwargs):
        self.model = model
        self.dataset = dataset
        self._batch_size = kwargs['batch_size'] if 'batch_size' in kwargs.keys() else 1

    def __len__(self):
        return len(self.dataset)

    def train_batch(self):
        '''
        produce one batch iteration
        :return:
        '''
        batch = self.dataset.next_batch(batch_size=self._batch_size, shuffle=True)
        print(batch)
        loss = 0
        return loss
    def train_epoch(self):
        loss = []
        while self.dataset.epochs_completed:
            loss.append(self.train_batch())
        return loss