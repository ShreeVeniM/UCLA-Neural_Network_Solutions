from sklearn.linear_model import LinearRegression

def model_trainn(Xtrain, ytrain):
    try:
        model = LinearRegression()
        model.fit(Xtrain, ytrain)
        return model
    except Exception as e:
        print(f"Error in model training: {e}")
        return None
