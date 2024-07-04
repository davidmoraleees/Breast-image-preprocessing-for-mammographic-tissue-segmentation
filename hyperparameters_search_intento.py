import os
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from tqdm import tqdm
from skimage import io
from skimage.measure import moments, moments_central, moments_normalized, moments_hu
from sklearn.metrics import classification_report

image_dir = 'Breast cancer diagnosis/INbreast/AllDICOMs_PNG/'

image_files_mlo = [f for f in os.listdir(image_dir) if 'R_ML' in f]
image_files_cc = [f for f in os.listdir(image_dir) if 'R_CC' in f]

def extract_features(image):
    hist, _ = np.histogram(image, bins=256, range=(0, 255))
    m = moments(image)
    centroid_row = m[0, 1] / m[0, 0]
    centroid_col = m[1, 0] / m[0, 0]
    mc = moments_central(image, center=(centroid_col, centroid_row))
    mn = moments_normalized(mc)
    mh = moments_hu(mn)
    features = np.hstack([hist, mh])
    return features

features_mlo = []
for file in tqdm(image_files_mlo):
    image_path = os.path.join(image_dir, file)
    image = io.imread(image_path)
    features = extract_features(image)
    features_mlo.append(features)

features_cc = []
for file in tqdm(image_files_cc):
    image_path = os.path.join(image_dir, file)
    image = io.imread(image_path)
    features = extract_features(image)
    features_cc.append(features)

features_mlo_array = np.array(features_mlo)
features_cc_array = np.array(features_cc)

labels_mlo = np.ones(len(features_mlo_array))
labels_cc = np.zeros(len(features_cc_array))

features_combined = np.vstack((features_mlo_array, features_cc_array))
labels_combined = np.hstack((labels_mlo, labels_cc))

X_train, X_test, y_train, y_test = train_test_split(features_combined, labels_combined, test_size=0.2, random_state=42)

selector = SelectKBest(score_func=f_classif, k=10)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

clf = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1)

grid_search.fit(X_train_selected, y_train)

print("Mejores parámetros:", grid_search.best_params_)

best_clf = grid_search.best_estimator_

y_pred = best_clf.predict(X_test_selected)
print(classification_report(y_test, y_pred))

selected_features_indices = selector.get_support(indices=True)
print("Índices de características seleccionadas:", selected_features_indices)
