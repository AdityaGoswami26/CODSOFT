import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MultiLabelBinarizer

# --- File Handling ---
def load_data( IMDb Movies India.csv):
    """
    Loads the CSV data, attempting various encodings.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: The loaded DataFrame, or None if loading fails.
    """
    encodings_to_try = ['utf-8', 'latin-1', 'cp1252']  # Common encodings

    for encoding in encodings_to_try:
        try:
            with open(file_path, encoding=encoding, errors='ignore') as f:
                data = pd.read_csv(f)
            print(f"Successfully read with encoding: {encoding}")
            return data
        except UnicodeDecodeError:
            print(f"Failed with encoding: {encoding}")
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            return None

    print("Failed to read with all encodings.")
    return None


# --- Feature Engineering ---
def preprocess_data(data):
    """
    Preprocesses the movie data by handling missing values and encoding categorical features.

    Args:
        data (pandas.DataFrame): The input DataFrame.

    Returns:
        pandas.DataFrame: The preprocessed DataFrame.
    """

    # Fill missing values for categorical columns with 'Unknown' instead of 0
    for col in ['Genre', 'Director', 'Actors']:
        if col in data.columns:
            data[col] = data[col].fillna('Unknown')

    if 'Genre' in data.columns:
        mlb = MultiLabelBinarizer()
        data = data.join(pd.DataFrame(mlb.fit_transform(data['Genre'].str.split('|')),
                                      columns=mlb.classes_,
                                      index=data.index))
        data.drop(columns=['Genre'], inplace=True, errors='ignore')

    for col in ['Director', 'Actors']:
        if col in data.columns:
            data = pd.get_dummies(data, columns=[col], prefix=col,
                                 dummy_na=False, drop_first=True)
    # For any other numeric columns, you can fill missing values with 0
    data = data.fillna(0)

    return data


# --- Model Training and Evaluation ---
def train_and_evaluate_model(X, y):
    """
    Trains a Random Forest Regressor model and evaluates its performance.

    Args:
        X (pandas.DataFrame): The feature matrix.
        y (pandas.Series): The target variable (ratings).
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')

    if hasattr(model, 'feature_importances_'):
        feature_importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        })
        feature_importances = feature_importances.sort_values(
            by='Importance', ascending=False)
        print("\nFeature Importances:\n", feature_importances)


# --- Main Execution ---
if __name__ == '__main__':
    file_path = 'IMDb Movies India.csv'  # replace with your filename/path
    data = load_data(file_path)

    if data is not None:
        data = preprocess_data(data)

        if 'Rating' in data.columns:
            y = data['Rating']
            X = data.drop(columns=['Rating'], errors='ignore')
            train_and_evaluate_model(X, y)
        else:
            print("Error: Target column 'Rating' not found in dataset.")
    else:
        print("Failed to load data.")
