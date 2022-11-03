class ExampleLoader():
    def __init__(self):
        self.arr = [1, 2, 3, 4, 5]
        
    def __getitem__(self, idx):
        return self.arr[idx]
    
    def __len__(self):
        return len(self.arr)
    
example_loader = ExampleLoader()
print(example_loader[2])
print(len(example_loader))