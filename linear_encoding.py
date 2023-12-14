import numpy as np
import os

dnn_fmaps_train = np.load("/users/sliao10/Desktop/CLPS1291/pca_feature_maps_training.npy", allow_pickle=True).item()
dnn_fmaps_test = np.load("/users/sliao10/Desktop/CLPS1291/pca_feature_maps_test.npy", allow_pickle=True).item()

print('Training DNN feature maps shape:')
print(dnn_fmaps_train['all_layers'].shape)
print('(Training image conditions × DNN feature maps PCA components)')

print('\nTest DNN feature maps shape:')
print(dnn_fmaps_test['all_layers'].shape)
print('(Test image conditions × DNN feature maps PCA components)')

# Train the encoding models
eeg_data_train_avg = np.mean(eeg_data_train['preprocessed_eeg_data'], 1)
eeg_data_train_avg = np.reshape(eeg_data_train_avg,
    (eeg_data_train_avg.shape[0], -1))
# dnn_fmaps_train['all_layers'] = dnn_fmaps_train['all_layers'][0:8000]
# eeg_data_train_avg = eeg_data_train_avg[0:8000]
reg = LinearRegression().fit(dnn_fmaps_train['all_layers'][:,:3000],
    eeg_data_train_avg)
pred_eeg_data_test = reg.predict(dnn_fmaps_test['all_layers'][:,:3000])
pred_eeg_data_test = np.reshape(pred_eeg_data_test,
    (-1, len(eeg_data_train['ch_names']), len(eeg_data_train['times'])))

# Test the encoding models
eeg_data_test_avg = np.mean(eeg_data_test['preprocessed_eeg_data'], 1)
encoding_accuracy = np.zeros((len(eeg_data_test['ch_names']),
    len(eeg_data_test['times'])))
for t in range(len(eeg_data_test['times'])):
    for c in range(len(eeg_data_test['ch_names'])):
        encoding_accuracy[c,t] = corr(pred_eeg_data_test[:,c,t],
            eeg_data_test_avg[:,c,t])[0]

print(np.max(np.mean(encoding_accuracy,0)))
# Plot the results
plt.figure()
plt.plot([-.2, .8], [0, 0], 'k--', [0, 0], [-1, 1], 'k--')
plt.plot(eeg_data_test['times'], np.mean(encoding_accuracy, 0));
plt.xlabel('Time (s)');
plt.xlim(left=-.2, right=.8)
plt.ylabel('Pearson\'s $r$');
plt.ylim(bottom=-.05, top=.7)
plt.title('Encoding accuracy');