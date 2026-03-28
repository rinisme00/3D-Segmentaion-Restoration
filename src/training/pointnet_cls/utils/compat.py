import sys
import types
from pathlib import Path


def find_project_root(start_dir=None):
    start = Path(start_dir or Path.cwd()).resolve()
    candidates = (start, *start.parents) if start.is_dir() else tuple(start.parents)
    for candidate in candidates:
        if (
            (candidate / "src" / "pointnet-master").is_dir()
            and (candidate / "data" / "classification").is_dir()
        ):
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
    tf.contrib.layers.xavier_initializer = tf1.initializers.glorot_uniform

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
        return tf.config.list_physical_devices("GPU")
    except Exception:
        return []


def resolve_device_name(tf, gpu_index):
    gpus = list_available_gpus(tf)
    if gpu_index is None or gpu_index < 0 or gpu_index >= len(gpus):
        return "/cpu:0"
    return "/gpu:{}".format(gpu_index)


def create_session_config(tf1):
    session_config = tf1.ConfigProto()
    session_config.gpu_options.allow_growth = True
    session_config.allow_soft_placement = True
    return session_config
