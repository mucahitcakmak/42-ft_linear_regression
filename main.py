import json

def checkModelFile(file_name="model.json"):
    try:
        with open (file_name, "r") as f:
            params = json.load(f)
            theta0, theta1 = params["theta0"], params["theta1"]
    
        try:
            theta0, theta1 = float(theta0), float(theta1)
        except (ValueError, TypeError):
            raise ValueError("Parameters 'theta0' and 'theta1' must be numeric (int or float).")
        
        return theta0, theta1
    except FileNotFoundError:
        raise FileNotFoundError("❌ File not found")
    except json.JSONDecodeError:
        raise ValueError("❌ Invalid JSON format")
    except Exception as e:
        raise RuntimeError(f"❌ Unexpected error: {e}")

def checkInput(input):
    try:
        input = float(input)
        return input
    except (ValueError, TypeError):
        raise ValueError("Input must be numeric (int or float).")

def main():
    theta0, theta1 = checkModelFile()

    mileage = checkInput(input("Enter your car mileage: "))

    print("Predict: ", (theta1 * mileage) + theta0) # normalizasyon hatasından sonra burası düzelecek değişiklik diğer dosyada yapılmalı
    
    
if __name__ == '__main__':
    main()