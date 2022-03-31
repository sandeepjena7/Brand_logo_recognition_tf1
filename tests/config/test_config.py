import os


def test_labelmap_path(test_read_yaml):
    config = test_read_yaml
    prediction_dir = config['prediction']["prediction_dir"]
    labelmap_dir = config['prediction']['labelmap_dir']
    labelmap_name = config['prediction']['labelmap_name']

    labelmap_path = os.path.join(prediction_dir, labelmap_dir,labelmap_name)
    assert os.path.exists(labelmap_path)


def test_ckpt_path(test_read_yaml):
    config = test_read_yaml
    prediction_dir = config['prediction']["prediction_dir"]
    ckpt_dir = config['prediction']["ckpt_dir"]
    model_name = config['prediction']["model_name"]

    ckpt_path = os.path.join(prediction_dir,ckpt_dir,model_name)
    assert os.path.exists(ckpt_path)