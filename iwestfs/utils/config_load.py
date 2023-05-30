
# ===================================================================
import sys
import re
import yaml
SimpleNamespace = type(sys.implementation)

def yaml_load(file='data.yaml', append_filename=False):
    """
    Load YAML data from a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.

    Returns:
        dict: YAML data and file name.
    """
    with open(file, errors='ignore', encoding='utf-8') as f:
        s = f.read()  # string

        # Remove special characters
        if not s.isprintable():
            s = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+', '', s)

        # Add YAML filename to dict and return
        return {**yaml.safe_load(s), 'yaml_file': str(file)} if append_filename else yaml.safe_load(s)

class IterableSimpleNamespace(SimpleNamespace):
    """
    Iterable SimpleNamespace class to allow SimpleNamespace to be used with dict() and in for loops
    """

    def __iter__(self):
        return iter(vars(self).items())

    def __str__(self):
        return '\n'.join(f'{k}={v}' for k, v in vars(self).items())

    def __getattr__(self, attr):
        name = self.__class__.__name__
        raise AttributeError(f"""
            '{name}' object has no attribute '{attr}'. This may be caused by a modified or out of date ultralytics
            'default.yaml' file.\nPlease update your code with 'pip install -U ultralytics' and if necessary replace
            {CFG_PATH} with the latest version from
            https://github.com/ultralytics/ultralytics/blob/main/ultralytics/yolo/cfg/default.yaml
            """)

    def get(self, key, default=None):
        return getattr(self, key, default)

def get_cfg(cfg, *cfgs):
    CFG_PATH = cfg
    CFG_DICT = yaml_load(CFG_PATH)

    for cfg in cfgs:
        CFG_PATH = cfg
        TEMP_DICT = yaml_load(CFG_PATH)
        CFG_DICT.update(TEMP_DICT)

    for k, v in CFG_DICT.items():
        if isinstance(v, str) and v.lower() == 'none':
            CFG_DICT[k] = None
    CFG_KEYS = CFG_DICT.keys()
    CFG = IterableSimpleNamespace(**CFG_DICT)
    return CFG

