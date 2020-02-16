import sys
sys.path.insert(0,"..")
import model_repository
import numpy as np
import utils
import io
import dill
from sklearn import metrics
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--num_trials', type=int, default=20, help='number of trials to loop over')
parser.add_argument('--training_size', type=int, default=None, help='how many examples to train with')
parser.add_argument('--eval_id', type=str, default="mPxwJFY376", help='evaluation ID')
args= parser.parse_args()

num_trials = args.num_trials
training_sizes = [args.training_size]
EVALUATION_ID = args.eval_id
if args.training_size is None:
    training_sizes = [1280, 640, 320, 160, 80, 40, 20, 10]

m_repo = model_repository.ModelRepository()
cifar_test_dataset_row = m_repo.get_dataset_by_name("cifar-10-zca-test")
cifar_train_dataset_row = m_repo.get_dataset_by_name("cifar-10-zca-train")
test_dataset = np.load(io.BytesIO(m_repo.get_dataset_data(str(cifar_test_dataset_row.uuid))))
train_dataset = np.load(io.BytesIO(m_repo.get_dataset_data(str(cifar_train_dataset_row.uuid))))

eval_obj = m_repo.get_evaluation(EVALUATION_ID, load_parents=True)
prediction_data = dill.loads(m_repo.get_evaluation_predictions_data(eval_obj.uuid))
y_test_full = test_dataset["y_test"]
y_train_full = train_dataset["y_train"]

K_train = utils.bytes_to_numpy(m_repo.get_checkpoint_kernel_data(eval_obj.checkpoint.uuid))
K_test = prediction_data["kernel"]

train_norms = np.sqrt(np.diag(K_train))[:, np.newaxis]
K_train = (K_train / train_norms)/train_norms.T

print('done loading kernel')

num_classes = 10
train_idx_0 = []
train_idx_1 = []
train_idx_2 = []
train_idx_3 = []
train_idx_4 = []
train_idx_5 = []
train_idx_6 = []
train_idx_7 = []
train_idx_8 = []
train_idx_9 = []
for i in range(len(y_train_full)):
    current_label = y_train_full[i]
    if current_label == 0:
        train_idx_0.append(i)
    elif current_label == 1:
        train_idx_1.append(i)
    elif current_label == 2:
        train_idx_2.append(i)
    elif current_label == 3:
        train_idx_3.append(i)
    elif current_label == 4:
        train_idx_4.append(i)
    elif current_label == 5:
        train_idx_5.append(i)
    elif current_label == 6:
        train_idx_6.append(i)
    elif current_label == 7:
        train_idx_7.append(i)
    elif current_label == 8:
        train_idx_8.append(i)
    elif current_label == 9:
        train_idx_9.append(i)

def sample_idx_class_balanced(training_size):
    train_subset_0_idx = np.random.choice(len(train_idx_0), training_size // num_classes, replace=False)
    train_subset = list(train_idx_0[i] for i in list(train_subset_0_idx))
    train_subset_1_idx = np.random.choice(len(train_idx_1), training_size // num_classes, replace=False)
    train_subset.extend(list(train_idx_1[i] for i in list(train_subset_1_idx)))
    train_subset_2_idx = np.random.choice(len(train_idx_2), training_size // num_classes, replace=False)
    train_subset.extend(list(train_idx_2[i] for i in list(train_subset_2_idx)))
    train_subset_3_idx = np.random.choice(len(train_idx_3), training_size // num_classes, replace=False)
    train_subset.extend(list(train_idx_3[i] for i in list(train_subset_3_idx)))
    train_subset_4_idx = np.random.choice(len(train_idx_4), training_size // num_classes, replace=False)
    train_subset.extend(list(train_idx_4[i] for i in list(train_subset_4_idx)))
    train_subset_5_idx = np.random.choice(len(train_idx_5), training_size // num_classes, replace=False)
    train_subset.extend(list(train_idx_5[i] for i in list(train_subset_5_idx)))
    train_subset_6_idx = np.random.choice(len(train_idx_6), training_size // num_classes, replace=False)
    train_subset.extend(list(train_idx_6[i] for i in list(train_subset_6_idx)))
    train_subset_7_idx = np.random.choice(len(train_idx_7), training_size // num_classes, replace=False)
    train_subset.extend(list(train_idx_7[i] for i in list(train_subset_7_idx)))
    train_subset_8_idx = np.random.choice(len(train_idx_8), training_size // num_classes, replace=False)
    train_subset.extend(list(train_idx_8[i] for i in list(train_subset_8_idx)))
    train_subset_9_idx = np.random.choice(len(train_idx_9), training_size // num_classes, replace=False)
    train_subset.extend(list(train_idx_9[i] for i in list(train_subset_9_idx)))
    return train_subset

for train_size in training_sizes:
    accuracies = []
    for trial in range(num_trials):
        idxs = sample_idx_class_balanced(train_size)
        K_train_sub = K_train[idxs, :][:, idxs]
        K_test_sub = K_test[:, idxs]
        alphas_sub = scipy.linalg.solve(K_train_sub, np.eye(10)[y_train_full[idxs]], sym_pos=True)
        logits_sub = K_test_sub.dot(alphas_sub)
        predictions = np.argmax(logits_sub, axis=1)
        acc = metrics.accuracy_score(predictions, y_test_full)
        accuracies.append(acc)
    accuracies = np.asarray(accuracies)*100
    print('training size: ', train_size)
    print('list of accuracies')
    print(accuracies)
    print(f'accuracies: {np.mean(accuracies)} ± {np.std(accuracies)}')