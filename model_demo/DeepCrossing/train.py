from utils import create_criteo_dataset
from model import DeepCrossing
import os
import sys
import tensorflow as tf
from tensorflow.keras import optimizers, losses
from sklearn.metrics import accuracy_score
import operators as sgx
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.realpath(os.path.join(root_dir,"train.txt"))
    feature_columns, (X_train, y_train), (X_test, y_test) = create_criteo_dataset(
        file_path, test_size=0.2
    )

    k = 32
    hidden_units = [256, 256]
    res_layer_num = 4
    sgx.create_enclave()
    model = DeepCrossing(feature_columns, k, hidden_units, res_layer_num)
    optimizer = optimizers.SGD(0.01)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)
    
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(train_dataset, epochs=10)
    logloss, auc = model.evaluate(X_test, y_test)
    print("logloss {}\nAUC {}".format(round(logloss, 2), round(auc, 2)))


    # 评估
    pre = model(X_test)
    pre = [1 if x > 0.5 else 0 for x in pre]
    print("Accuracy: ", accuracy_score(y_test, pre))
