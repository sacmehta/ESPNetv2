
#============================================
__author__ = "Sachin Mehta"
__license__ = "MIT"
__maintainer__ = "Sachin Mehta"
#============================================

class MyLRScheduler(object):
    '''
    CLass that defines cyclic learning rate that decays the learning rate linearly till the end of cycle and then restarts
    at the maximum value.
    '''
    def __init__(self, initial=0.1, cycle_len=5, steps=[51, 101, 131, 161, 191, 221, 251, 281], gamma=2):
        super(MyLRScheduler, self).__init__()
        assert len(steps) > 1, 'Please specify step intervals.'
        self.min_lr = initial # minimum learning rate
        self.m = cycle_len
        self.steps = steps
        self.warm_up_interval = 1 # we do not start from max value for the first epoch, because some time it diverges
        self.counter = 0
        self.decayFactor = gamma # factor by which we should decay learning rate
        self.count_cycles = 0
        self.step_counter = 0
        self.stepping = True
        print('Using Cyclic LR Scheduler with warm restarts')

    def get_lr(self, epoch):
        if epoch%self.steps[self.step_counter] == 0 and epoch > 1 and self.stepping:
            self.min_lr = self.min_lr / self.decayFactor
            self.count_cycles = 0
            if self.step_counter < len(self.steps) - 1:
                self.step_counter += 1
            else:
                self.stepping = False
        current_lr = self.min_lr
        # warm-up or cool-down phase
        if self.count_cycles < self.warm_up_interval:
            self.count_cycles += 1
            # We do not need warm up after first step.
            # so, we set warm up interval to 0 after first step
            if self.count_cycles == self.warm_up_interval:
                self.warm_up_interval = 0
        else:
            #Cyclic learning rate with warm restarts
            # max_lr (= min_lr * step_size) is decreased to min_lr using linear decay before
            # it is set to max value at the end of cycle.
            if self.counter >= self.m:
                self.counter = 0
            current_lr = round((self.min_lr * self.m) - (self.counter * self.min_lr), 5)
            self.counter += 1
            self.count_cycles += 1
        return current_lr


if __name__ == '__main__':
    lrSched = MyLRScheduler(0.1)#MyLRScheduler(0.1, 5, [51, 101, 131, 161, 191, 221, 251, 281, 311, 341, 371])
    max_epochs = 300
    for i in range(max_epochs):
        print(i, lrSched.get_lr(i))
