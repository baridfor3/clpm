# -*- coding: utf-8 -*-
# code warrior: Barid

import contextlib
import tensorflow as tf
import sys
import os

cwd = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(cwd, os.pardir)))
os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"  # fp16 training
tf.config.set_soft_device_placement(True)
tf.config.optimizer.set_jit(False)

import argparse
@contextlib.contextmanager
def config_options(options):
    old_opts = tf.config.optimizer.get_experimental_options()
    tf.config.optimizer.set_experimental_options(options)
    try:
        yield
    finally:
        tf.config.optimizer.set_experimental_options(old_opts)


options = {
    "layout_optimizer": True,
    "constant_folding": True,
    "shape_optimization": True,
    "remapping": True,
    "arithmetic_optimization": True,
    "dependency_optimization": True,
    "loop_optimization": True,
    "function_optimization": True,
    "debug_stripper": True,
    "disable_model_pruning": True,
    "scoped_allocator_optimization": True,
    "pin_to_host_optimization": True,
    "implementation_selector": True,
    "auto_mixed_precision": True,
    "disable_meta_optimizer": True,
    "min_graph_nodes": True,
}
config_options(options)



from UNIVERSAL.app import basic_app, multilingual_configuration
import initialization
import CLPM_MLM

profile = {
"UNIVERSAL_PATH":"/home/vivalavida/workspace/alpha/UNIVERSAL",
"project_path": os.getcwd(),
"corpora_path": "/home/vivalavida/massive_data/data/wiki/EnDeHi_6k",
"GPU":2
}


parameters = multilingual_configuration.get_parameters(profile)
config_builder = multilingual_configuration.get_config_builder(parameters)
# ---------CrossInit--------------------------
# config_builder["model_class"] = initialization.necXLM
# config_builder["preprocess_fn"] = initialization.nec

# #------------------training--------------
parameters["domain_index"] = initialization.get_domain_index()
parameters["CLPM_warmup"] = 50000
parameters["alternation_ratio"] = 0.5
config_builder["model_class"] = CLPM_MLM.CLPM_MLM

if __name__ == "__main__":
    app = basic_app.APP(config_builder,parameters)
    app.compile()
    app.trainer()