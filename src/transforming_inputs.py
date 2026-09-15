import pandas as pd

class TransformInputs:
    def __init__(self, age_imputer, encoder, scaler):
        """Initialize with the fitted transformers from the pickle file."""
        self.age_imputer = age_imputer
        self.encoder = encoder
        self.scaler = scaler

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies the training-time transformations to incoming inference data."""
        # Work on a copy to avoid SettingWithCopyWarnings
        input_df = df.copy()
        
        # 1. Age imputation
        input_df[['Age']] = self.age_imputer.transform(input_df[['Age']])

        # 2. One Hot Encoding
        cat_cols = ['Sex', 'Embarked']
        encoded_cats = self.encoder.transform(input_df[cat_cols])
        
        encoded_df = pd.DataFrame(
            encoded_cats,
            columns=self.encoder.get_feature_names_out(cat_cols)
        )

        # 3. Remove original categorical columns
        input_df = input_df.drop(columns=cat_cols)

        # 4. Add encoded columns
        input_df = pd.concat(
            [input_df.reset_index(drop=True), encoded_df],
            axis=1
        )

        # 5. Scaling
        input_df[['Age', 'Fare']] = self.scaler.transform(input_df[['Age', 'Fare']])

        return input_df