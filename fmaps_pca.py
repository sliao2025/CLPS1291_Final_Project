import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
save_dir = "/users/sliao10/Desktop/CLPS1291"
seed = 20200220

feats = []
feats_all = []
fmaps_train = {}
fmaps_dir = os.path.join(save_dir, "vgg16_training_fmaps")
fmaps_list = os.listdir(fmaps_dir)
fmaps_list.sort()
print(len(fmaps_list))

#Loading the feature maps
print(fmaps_list)

for f, fmaps in enumerate(fmaps_list):
    print(f)
    fmaps_data = np.load(os.path.join(fmaps_dir, fmaps), allow_pickle=True).item()
    all_layers = fmaps_data.keys()
    for l, dnn_layer in enumerate(all_layers):
        if l == 0:
            feats = np.reshape(fmaps_data[dnn_layer], -1)
        else:
            feats = np.append(feats, np.reshape(fmaps_data[dnn_layer], -1))
    feats_all.append(feats)

layer_names = ['all_layers']
fmaps_train[layer_names[0]] = np.asarray(feats_all)

print(fmaps_train['all_layers'].shape)


# Standardize the data
scaler = []
for l, dnn_layer in enumerate(layer_names):
    print('scale')
    scaler.append(StandardScaler())
    scaler[l].fit(fmaps_train[dnn_layer])
    fmaps_train[dnn_layer] = scaler[l].transform(fmaps_train[dnn_layer])

print(fmaps_train['all_layers'].shape)

# Apply Non-Linear PCA
pca = []
for l, dnn_layer in enumerate(layer_names):
    print('pca')
    pca.append(KernelPCA(n_components=3000, kernel='poly',
		degree=4, random_state=seed))
    pca[l].fit(fmaps_train[dnn_layer])
    fmaps_train[dnn_layer] = pca[l].transform(fmaps_train[dnn_layer])

print(fmaps_train['all_layers'].shape)


# Save the downsampled feature maps
file_name = 'pca_feature_maps_training'
np.save(os.path.join(save_dir,"vgg16_pca_files", file_name), fmaps_train)




feats = []
feats_all = []
fmaps_test = {}
fmaps_dir = os.path.join(save_dir, "vgg16_test_fmaps")
fmaps_list = os.listdir(fmaps_dir)
fmaps_list.sort()
print(len(fmaps_list))

#Loading the feature maps
print(fmaps_list)

for f, fmaps in enumerate(fmaps_list):
    print(f)
    fmaps_data = np.load(os.path.join(fmaps_dir, fmaps), allow_pickle=True).item()
    all_layers = fmaps_data.keys()
    for l, dnn_layer in enumerate(all_layers):
        if l == 0:
            feats = np.reshape(fmaps_data[dnn_layer], -1)
        else:
            feats = np.append(feats, np.reshape(fmaps_data[dnn_layer], -1))
    feats_all.append(feats)

layer_names = ['all_layers']
fmaps_test[layer_names[0]] = np.asarray(feats_all)

print(fmaps_test['all_layers'].shape)


# Standardize the data
for l, dnn_layer in enumerate(layer_names):
    print('scale')
    fmaps_test[dnn_layer] = scaler[l].transform(fmaps_test[dnn_layer])

print(fmaps_test['all_layers'].shape)

# Apply Non-Linear PCA
for l, dnn_layer in enumerate(layer_names):
    print('pca')
    fmaps_test[dnn_layer] = pca[l].transform(fmaps_test[dnn_layer])

print(fmaps_test['all_layers'].shape)


# Save the downsampled feature maps
file_name = 'pca_feature_maps_test'
np.save(os.path.join(save_dir,"vgg16_pca_files", file_name), fmaps_test)