from toy_duration_predictor.preprocess import mssv
from toy_duration_predictor.load import load_and_test
import toy_duration_predictor.upload as ul


def preprocessing():
    mssv_path = "/mnt/d/dataset/004.다화자 가창 데이터"
    mssv_preprocessed_path = "/mnt/d/dataset/mssv_preprocessed_duration"
    mssv.preprocess_dataset(mssv_path, mssv_preprocessed_path)


if __name__ == "__main__":
    # preprocessing()
    ul.upload_models_to_hub()
    load_and_test()
