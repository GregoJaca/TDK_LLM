from dataclasses import dataclass
from typing import Any, Dict
import json

import config as GLOBAL_CONFIG_MODULE
GLOBAL_CONFIG = GLOBAL_CONFIG_MODULE.CONFIG


@dataclass
class LyapunovConfig:
    cfg: Dict[str, Any]

    @classmethod
    def from_global(cls):
        return cls(GLOBAL_CONFIG.get("lyapunov", {}))

    def get(self, key, default=None):
        return self.cfg.get(key, default)
