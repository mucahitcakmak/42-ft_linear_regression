import json
import os

def checkModelFile(file_name="model.json"):
    try:
        if (os.path.exists(file_name) == False):
            params = {"theta0": 0, "theta1": 0}
            with open("model.json", "w") as f:
                json.dump(params, f)

        with open (file_name, "r") as f:
            params = json.load(f)
            theta0, theta1 = params["theta0"], params["theta1"]
    
        try:
            theta0, theta1 = float(theta0), float(theta1)
        except (ValueError, TypeError):
            raise ValueError("Parameters 'theta0' and 'theta1' must be numeric (int or float).")
        
        return theta0, theta1
    except json.JSONDecodeError:
        raise ValueError("❌ Invalid JSON format")
    except Exception as e:
        raise RuntimeError(f"❌ Unexpected error: {e}")

def checkInput(input):
    try:
        input = float(input)
        return input
    except (ValueError, TypeError):
        raise ValueError("❌ Unexpected error: Input must be numeric (int or float).")

def main():
    try:
        theta0, theta1 = checkModelFile()

        mileage = checkInput(input("Enter your car mileage: "))

        estimated_price = (theta1 * mileage) + theta0
        print("Predict: ", round(estimated_price, 2))
    except Exception as e:
        print(e)
    
    
if __name__ == '__main__':
    main()