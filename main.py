# main.py

from modules import config, data_loader, feature_extractor, classifier

def main():
    # Step 1: Load Data
    print("Creating data generators...")
    train_generator, val_generator, test_generator = data_loader.create_data_generators(config.DATASET_DIR, config.BATCH_SIZE)

    # Step 2: Feature Extraction using CNN
    features_extractor = feature_extractor.FeatureExtractor()
    X_train, y_train = features_extractor.extract_features(train_generator, 'train')
    X_val, y_val = features_extractor.extract_features(val_generator, 'validation')
    X_test, y_test = features_extractor.extract_features(test_generator, 'test')

    # Step 3: Prepare Data for Classifier
    classifier_obj = classifier.ClassicalClassifier()
    X_train_pca, X_val_pca, X_test_pca = classifier_obj.prepare_data(X_train, X_val, X_test)

    # Step 4: Train Classifier
    classifier_obj.train(X_train_pca, y_train)

    # Step 5: Evaluate Classifier
    classifier_obj.evaluate(X_test_pca, y_test)

if __name__ == "__main__":
    main()
