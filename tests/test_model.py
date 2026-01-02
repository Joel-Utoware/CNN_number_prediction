import pytest
from pathlib import Path
from model.load_model import get_best_model, predict_digit

@pytest.fixture(scope="session")
def model():
    return get_best_model()

def test_model_loading():
    assert model is not None

def test_correct_answer(model):
    image_path = Path(__file__).parent / "data" / "num_1.png"
    prediction, probs = predict_digit(image_path, model=model)
    assert prediction == 1

def test_predict_digit_invalid_path_raises_error(model):
    fake_path = Path("this_file_does_not_exist.png")
    with pytest.raises(FileNotFoundError):
        predict_digit(fake_path, model)

def test_predict_digit_wrong_type_raises_error(model):
    with pytest.raises((TypeError, AttributeError)):
        predict_digit(12345, model)

def test_predict_digit_non_image_file_raises_error(model):
    bad_file = Path(__file__).parent / "data" / "not_an_image.txt"

    with pytest.raises(Exception):
        predict_digit(bad_file, model)