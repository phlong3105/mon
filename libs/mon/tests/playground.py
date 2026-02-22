#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Playground."""
from dataclasses import dataclass, field

# noinspection PyUnusedImports
import mon
from mon import Path
import aenum

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


"""
output_dir = "/Volumes/ssd_01/10_workspace/11_code/mon/projects/enhance/run/predict/zerodce/zerodce/dicm"
dirname = "pred"
subdirname = ""
src_path = "/Volumes/ssd_01/10_workspace/11_code/mon/projects/enhance/data/dicm/test/image/01.jpg"
print(mon.resolve_save_dir(output_dir, dirname, subdirname, src_path, keep_subdirs=False, near_src=False))
print(mon.resolve_save_dir(output_dir, dirname, subdirname, src_path, keep_subdirs=True,  near_src=False))
print(mon.resolve_save_dir(output_dir, dirname, subdirname, src_path, keep_subdirs=False, near_src=True))
print(mon.resolve_save_dir(output_dir, dirname, subdirname, src_path, keep_subdirs=True,  near_src=True))
"""


"""
weights1 = mon.Path("/Volumes/ssd_01/10_workspace/11_code/mon/zoo/cv/classify/mobileone/mobileone_s0/imagenet1k_v1/mobileone_s0_imagenet1k_v1.pth.tar")
weights2 = mon.Path("/Volumes/ssd_01/10_workspace/11_code/mon/projects/enhance/run/train/clode/clode/clode_siceme/last.pt")
root = mon.ZOO_ROOT
print(weights1.unique_path_from(root).truncate())
print(weights2.unique_path_from(root).truncate())
"""


"""
config_ctx = mon.ConfigContext(root=current_dir, config_file="alexnet_v2.yaml")
config_ctx.config_for("predict", prompt=True)
config_ctx.log_summary(full=True)
"""


"""
@dataclass
class Foo:
    a: int
    _a: int = field(init=False, repr=False)

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, value):
        self._a = value

foo = Foo(a=3)
bar = Foo(a=2)
print(foo.a)
print(bar.a)
"""


class DefaultMultiStrEnum(str, aenum.MultiValueEnum):

    # 1. Override how it prints natively
    def __str__(self):
        return self.value

    # 2. Override how it behaves inside f-strings
    def __format__(self, format_spec):
        return str.__format__(self.value, format_spec)

    @classmethod
    def _missing_(cls, value):
        # 1. If no value is passed (None), return the first member
        if value is None:
            return list(cls)[0]

        # 3. Otherwise, it's an invalid extension
        aenum.extend_enum(cls, value, value)
        return cls(value)

    # --- Retrieval ---
    @classmethod
    def names(cls) -> list[str]:
        """Returns a list of all enum member names."""
        return [member.name for member in cls]

    @classmethod
    def values(cls) -> list[str]:
        """Returns a list of all primary enum member values."""
        return [member.value for member in cls]


class ConfigExtension(DefaultMultiStrEnum):
    """Enum for configuration file extensions."""

    # Because YAML is first, it automatically becomes the default fallback!
    YAML = ".yaml", ".yml", "default"
    CFG  = ".cfg", ".config"
    JSON = ".json"
    TXT  = ".txt"


# --- PROOF YOUR METACLASS LOGIC STILL WORKS ---

# 1. Passing None returns the first member
print(ConfigExtension(None))        # Output: .yaml

# 2. Passing "default" returns the DEFAULT attribute
print(ConfigExtension("default"))   # Output: .yaml

# 3. Standard alias mapping still works natively
print(ConfigExtension(".yml"))      # Output: .yaml

file_path = f"settings{ConfigExtension('default')}"
print(file_path)                   # Output: settings.yaml

print(ConfigExtension(".asd"))
