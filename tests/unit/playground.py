import importlib

module = importlib.import_module("one.cfg.zerodce_lime")
print(module.__file__)
print(module.data)
