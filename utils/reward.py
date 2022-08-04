class RewardBuffer():
    def __init__(self):
        self.maxSize = 300
        self.data = []
    
    def append(self, val):
        if len(self.data) > self.maxSize:
            self.data.pop()
        
        self.data.append(val)
    
    def info(self):
        for el in self.data:
            print(el)