import logging
from src.data_loading import load_data
from src.features import engineer_features, scale_features, create_interaction_terms
from src.model_train import model_trainn
from src.model_evaluation import model_evaluate
from src.data_partition import data_partitioning
from src.utils import setup_logging

setup_logging()

def main():
    logging.info('Starting main function')
    try:
        # Define the folder to store charts
        chart_folder = 'output_charts'

        # Load data
        df = load_data('src/dataset/Admission.csv')
        logging.info('Data loaded successfully')

        # Engineer features
        df = engineer_features(df)
        logging.info('Feature engineering completed')

        # Scale numeric features
        numeric_columns = ['GRE_Score', 'TOEFL_Score', 'CGPA']  # Example numeric columns
        df = scale_features(df, numeric_columns)
        logging.info('Scaling of numeric features completed')

        # Create interaction terms
        interaction_pairs = [('GRE_Score', 'CGPA'), ('TOEFL_Score', 'CGPA')]  # Example pairs
        df = create_interaction_terms(df, interaction_pairs)
        logging.info('Creation of interaction terms completed')

        # Separate features and target
        X = df.drop('Admit_Chance', axis=1)
        y = df['Admit_Chance']

        # Partition data
        Xtrain, Xtest, ytrain, ytest = data_partitioning(X, y)
        logging.info('Data partitioning completed')

        # Model training
        model = model_trainn(Xtrain, ytrain)
        logging.info('Model training completed')

        # Model evaluation
        mse, r2 = model_evaluate(model, Xtest, ytest, folder=chart_folder)
        logging.info(f"Model evaluation results: MSE={mse}, R2={r2}")

    except Exception as e:
        logging.error("An error occurred in main: %s", e)
        raise

if __name__ == '__main__':
    main()
