from sklearn.model_selection import train_test_split

def data_partitioning(X, y, test_size=0.2, random_state=42):
    try:
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return Xtrain, Xtest, ytrain, ytest
    except Exception as e:
        print(f"Error in data partitioning: {e}")
        return None, None, None, None
