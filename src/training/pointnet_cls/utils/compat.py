import sys
import types
from pathlib import Path


def find_project_root(start_dir=None):
    start = Path(start_dir or Path.cwd()).resolve()
    candidates = (start, *start.parents) if start.is_dir() else tuple(start.parents)
    for candidate in candidates:
        if (candidate / "src" / "training" / "pointnet_cls").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate project root from {}".format(start))


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)
POINTNET_ROOT = PROJECT_ROOT / "src" / "pointnet-master"


def ensure_pointnet_paths():
    for path in (POINTNET_ROOT, POINTNET_ROOT / "models", POINTNET_ROOT / "utils"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    return POINTNET_ROOT


def load_tensorflow():
    import tensorflow as tf

    tf1 = tf.compat.v1
    tf1.disable_eager_execution()

    if not hasattr(tf, "contrib"):
        tf.contrib = types.SimpleNamespace()
    if not hasattr(tf.contrib, "layers"):
        tf.contrib.layers = types.SimpleNamespace()
    tf.contrib.layers.xavier_initializer = tf1.glorot_uniform_initializer
    tf.contrib.layers.xavier_initializer_conv2d = tf1.glorot_uniform_initializer

    tf.constant_initializer = tf1.constant_initializer
    tf.truncated_normal_initializer = tf1.truncated_normal_initializer
    tf.random_normal_initializer = tf1.random_normal_initializer
    tf.glorot_uniform_initializer = tf1.glorot_uniform_initializer

    tf.get_variable = tf1.get_variable
    tf.variable_scope = tf1.variable_scope
    tf.placeholder = tf1.placeholder
    tf.train = tf1.train
    tf.nn.max_pool = tf1.nn.max_pool
    tf.global_variables_initializer = tf1.global_variables_initializer
    tf.Session = tf1.Session
    tf.ConfigProto = tf1.ConfigProto
    tf.summary = tf1.summary
    tf.add_to_collection = tf1.add_to_collection
    tf.truncated_normal_initializer = tf1.truncated_normal_initializer
    tf.to_int64 = lambda x: tf.cast(x, tf.int64)
    tf.device = tf1.device
    tf.cond = tf1.cond
    return tf, tf1


def list_available_gpus(tf):
    try:
        # Try TF2 way first
        gpus = tf.config.list_physical_devices("GPU")
        if gpus: return gpus
    except Exception:
        pass
    
    try:
        # Try TF1 way
        from tensorflow.python.client import device_lib
        local_device_protos = device_lib.list_local_devices()
        return [x for x in local_device_protos if x.device_type == 'GPU']
    except Exception:
        return []


def resolve_device_name(tf, gpu_index):
    gpus = list_available_gpus(tf)
    if not gpus:
        return "/cpu:0"
    
    # If index is invalid, default to the first available GPU
    idx = gpu_index if (gpu_index is not None and 0 <= gpu_index < len(gpus)) else 0
    return "/gpu:{}".format(idx)


def create_session_config(tf1):
    session_config = tf1.ConfigProto()
    session_config.gpu_options.allow_growth = True
    session_config.allow_soft_placement = True
    return session_config
