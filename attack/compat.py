# -*- coding: utf-8 -*-
"""
ByteTrack(2021) 과 최신 numpy 사이의 호환 계층

ByteTrack 코드는 numpy 1.20 에서 폐기 예고되고 1.24 에서 제거된 별칭을 쓴다.

    yolox/tracker/byte_tracker.py:18   np.asarray(tlwh, dtype=np.float)
    yolox/tracker/matching.py:61,66,67,121,124,127
    yolox/motdt_tracker/... (우리는 쓰지 않지만 import 될 수 있음)

numpy 2.x 에서는 AttributeError 가 난다.

    AttributeError: module 'numpy' has no attribute 'float'

**ByteTrack 원본을 수정하지 않기 위해** 여기서 별칭을 되살린다.
이 모듈은 yolox 를 import 하기 전에 먼저 import 해야 한다.

numpy 가 구버전이라 별칭이 이미 있으면 아무것도 하지 않는다(랩 PC 등).
"""

import numpy as np

# ByteTrack 이 실제로 쓰는 것만 되살린다 (grep 으로 확인: float, int, bool)
_ALIASES = {
    'float': float,
    'int': int,
    'bool': bool,
}

_restored = []
for _name, _py in _ALIASES.items():
    if not hasattr(np, _name):
        setattr(np, _name, _py)
        _restored.append(_name)


# motmetrics 1.4.0 이 쓰는 np.asfarray 도 numpy 2.0 에서 제거됐다.
#   motmetrics/distances.py:117  objs = np.asfarray(objs)
if not hasattr(np, 'asfarray'):
    def _asfarray(a, dtype=np.float64):
        return np.asarray(a, dtype=dtype)
    np.asfarray = _asfarray
    _restored.append('asfarray')


def restored_aliases():
    """되살린 별칭 목록. 디버깅·기록용."""
    return list(_restored)
