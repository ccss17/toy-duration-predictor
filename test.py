from toy_duration_predictor.preprocess import mssv
import toy_duration_predictor.train_fastai as train_fastai


def preprocessing():
    mssv_path = "/mnt/d/dataset/004.다화자 가창 데이터"
    mssv_preprocessed_path = "/mnt/d/dataset/mssv_preprocessed_duration"
    mssv.preprocess_dataset(mssv_path, mssv_preprocessed_path)


def test_train():
    train_fastai.test_train()


if __name__ == "__main__":
    # preprocessing()
    test_train()
