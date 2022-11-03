class Car:
    def __init__(self, model, year):
        self.model = model
        self.year = year
        
    def get_model(self):
        return self.model
    
    def get_year(self):
        return self.year


carA = Car('Audi', 2019)
print(carA.get_model())
print(carA.get_year())
    