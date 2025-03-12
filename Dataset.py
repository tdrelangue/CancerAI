from random import seed, shuffle
import pickle
import os
import numpy as np
import pandas as pd

class Dataset():
    NAValue = -1

    def __init__(self, patients=None, trainingRatio=None, validationRatio=None, seedValue=None, 
                pca_params=None, labels=None, target_features=None, pklLocation=None, test_mode=False):
        """
        Initialize the Dataset object in three different ways:
        1. If `pklLocation` is provided, load data from a pickle file.
        2. If `test_mode=True`, download GEO test data and generate labels.
        3. Otherwise, manually set `patients` and split into train/val/test sets.
        """
        self.seed = seedValue
        self.pca_params = pca_params

        if test_mode and not pklLocation:
            # Load test dataset (real 450K methylation data from GEO)
            self.load_test_data()
            self.seed = seedValue
            self.trainingSet, self.validationSet, self.testingSet = self.divideIntoSets(
                group=self.patients, labels=self.labels, target_features=target_features, 
                trainingRatio=trainingRatio, validationRatio=validationRatio, seedValue=seedValue
            )

        elif pklLocation:
            # Load dataset from pickle file
            self.load_from_pkl(pklLocation)

        elif patients is not None:
            # Initialize with provided patient data
            self.patients = patients.to_numpy()
            self.labels = labels.to_numpy()
            self.column_names = patients.columns  # ✅ Save column names before conversion
            self.column_names.extend(list(labels.columns))
            self.trainingSet, self.validationSet, self.testingSet = self.divideIntoSets(
                group=self.patients, labels=self.labels, target_features=target_features, 
                trainingRatio=trainingRatio, validationRatio=validationRatio, seedValue=seedValue
            )
        else:
            raise ValueError("You must provide either `patients`, `pklLocation`, or enable `test_mode`.")

        self.data = np.column_stack([self.patients, self.labels])

    def load_from_pkl(self, pklLocation):
        """Loads dataset from a pickle file."""
        try:
            with open(pklLocation, 'rb') as file:
                data = pickle.load(file)
                self.trainingSet, self.validationSet, self.testingSet,self.column_names = data["training"], data["validation"], data["testing"], data["columns"]
        except Exception as e:
            print(f"Error loading pickle file: {e}")
            self.trainingSet, self.validationSet, self.testingSet = None, None, None
        self.patients= np.vstack([self.trainingSet[0], self.validationSet[0], self.testingSet[0]])
        self.labels= np.vstack([self.trainingSet[1], self.validationSet[1], self.testingSet[1]])

    def load_test_data(self):
        """Loads test methylation data from GEO and generates random labels."""
        import GEOparse
        print("Running in test mode: Downloading real 450K methylation data...")

        # Download GEO dataset (example: GSE68777)
        gse = GEOparse.get_GEO("GSE68777", destdir="./")

        # Extract beta values (methylation data)
        df = gse.pivot_samples("VALUE")
        print(df.head())
        # ✅ Replace missing values (NaN) with the defined `NAValue`
        df = df.fillna(self.NAValue)

        # Convert DataFrame to NumPy array
        self.patients = df.to_numpy()  # Only numerical values

        # Number of samples (lines in original df)
        num_samples = self.patients.shape[0]

        # Generate 1% probability for '1', 99% for '0'
        random_testis = np.random.choice([0, 1], size=(num_samples,), p=[1-(1/270), 1/270])
        random_kidney = np.random.choice([0, 1], size=(num_samples,), p=[1-0.023, 0.023])
        random_prostate = np.random.choice([0, 1], size=(num_samples,), p=[1-(1/350), 1/350])

        # Store labels as a NumPy array
        self.labels = np.column_stack([random_testis, random_kidney, random_prostate])
        self.column_names = list(df.columns)
        self.column_names.extend(["Testis", "Kidney", "Prostate"]) # ✅ Save column names before conversion

    def divideIntoSets(self, group, labels, target_features, trainingRatio, validationRatio, seedValue):
        from sklearn.model_selection import train_test_split
        seed(seedValue)    
        if not labels is None:
            Y=labels
            X=group
        elif not target_features is None:
            # Extract Y (labels)
            Y = group[target_features].to_numpy()  # Convert to NumPy array
            # Extract X (features) - all other columns except the target columns
            X = group.drop(columns=target_features).to_numpy()
        else:
            raise ValueError("No target features given")

        # Step 1: Split into training and temp (validation + test)
        X_trainSet, X_temp, Y_trainSet, Y_temp = train_test_split(
            X, Y, train_size=trainingRatio, random_state=seedValue
        )

        # Step 2: Compute remaining ratio for validation and test
        remaining_ratio = 1 - trainingRatio
        validation_adjusted_ratio = validationRatio / remaining_ratio  # Re-adjust for second split

        # Step 3: Split temp into validation and test
        X_valSet, X_testSet, Y_valSet, Y_testSet = train_test_split(
            X_temp, Y_temp, train_size=validation_adjusted_ratio, random_state=seedValue
        )
        # Print results
        return (X_trainSet, Y_trainSet), (X_valSet, Y_valSet), (X_testSet, Y_testSet)

    def save(self, pickleLocation=None, processingParamsLocation=None):
        pickleDic = {"training" : self.trainingSet,
                    "validation" : self.validationSet,
                    "testing" : self.testingSet,
                    "columns" : self.column_names}

        if not pickleLocation is None:
            os.makedirs(os.path.dirname(pickleLocation), exist_ok=True)
            with open(pickleLocation, 'wb') as saveTo:
                pickle.dump(pickleDic, saveTo)

        if not processingParamsLocation is None:
            os.makedirs(os.path.dirname(processingParamsLocation), exist_ok=True)
            with open(processingParamsLocation, 'wb') as saveTo:
                pickle.dump(self.pca_params, saveTo)

    def __str__(self):
        trainingSet=np.column_stack([self.trainingSet[0], self.trainingSet[1]])
        validationSet=np.column_stack([self.validationSet[0], self.validationSet[1]])
        testingSet=np.column_stack([self.testingSet[0], self.testingSet[1]])

        # Convert training, validation, and testing sets into temporary Pandas DataFrames (for display only)
        trainingSet_df = pd.DataFrame(trainingSet, columns=self.column_names) if trainingSet is not None else None
        validationSet_df = pd.DataFrame(validationSet, columns=self.column_names) if validationSet is not None else None
        testingSet_df = pd.DataFrame(testingSet, columns=self.column_names) if testingSet is not None else None

        return (
            f"Dataset Summary:\n"
            f"  - Training Set: {trainingSet.shape if trainingSet is not None else 'Not Set'}\n"
            f"{trainingSet_df.head()}\n"
            f"  - Validation Set: {validationSet.shape if validationSet is not None else 'Not Set'}\n"
            f"{validationSet_df.head()}\n"
            f"  - Testing Set: {testingSet.shape if testingSet is not None else 'Not Set'}\n"
            f"{testingSet_df.head()}"
        )


