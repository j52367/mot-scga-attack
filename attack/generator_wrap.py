# -*- coding: utf-8 -*-
"""
generator 블록별 feature 추출 래퍼

SCGA 의 핵심 기여인 'generator 내부 일관성' 손실을 계산하려면
generator 의 블록별 중간 feature 가 필요하다. 그런데 공개된 generator 구현은
어느 것도 블록 리스트를 반환하지 않는다.

    generator/resnet.py     GeneratorResnet.forward(x, feat=True)
                            -> (output, features)  features 는 res_blocks 출력 '텐서 하나'
    generator/resnet_gap.py ResnetGenerator.forward(input)        feat 인자 없음
    generator/resnet_nat.py StableGeneratorResnet.forward(input)  feat 인자 없음
    generator/resnet_gap.py ...forward_features()  텐서 하나 + .detach() (gradient 끊김)

반면 학습 스크립트(train_BIA_ours.py)는 블록 리스트를 가정한다.

    taus = [0.6 + 0.05 * i for i in range(6)]   # 주석: "thresholds for 6 blocks"
    for j in range(len(feats_teacher[1:]) - 4):
        hinge(feats_teacher[1:][j], feats_student[1:][j], threshold=taus[j])

feats 가 텐서일 경우 [1:] 는 배치 차원을 자르게 되어 의도와 다르게 동작한다.

이 래퍼는 **generator/ 원본을 수정하지 않고** forward hook 으로 블록 출력을 모아
리스트로 돌려준다. hook 은 호출할 때마다 등록하고 즉시 제거하므로
deepcopy(teacher 생성)나 apply(초기화)에 영향을 주지 않는다.
"""

import torch.nn as nn


# GeneratorResnet 기준 블록 순서. 6개라 taus 의 6개와 대응된다.
DEFAULT_BLOCKS = ('block1', 'block2', 'block3',
                  'res_blocks', 'upsample1', 'upsample2')


class GeneratorWithFeatures(nn.Module):
    """
    사용법은 SCGA 와 동일하다.

        netG = build_generator(...)
        netG = GeneratorWithFeatures(netG)
        out, feats = netG(x, feat=True)      # feats: 블록별 텐서 리스트

    Args:
        netG        : generator/ 에서 만든 원본 모듈
        block_names : hook 을 걸 속성 이름. None 이면 DEFAULT_BLOCKS 중 존재하는 것만.
    """

    def __init__(self, netG, block_names=None):
        super().__init__()
        self.netG = netG
        names = DEFAULT_BLOCKS if block_names is None else block_names
        # 모듈 객체가 아니라 '이름'을 들고 있어야 deepcopy 후에도 올바른 사본을 가리킨다
        self.block_names = [n for n in names if isinstance(getattr(netG, n, None), nn.Module)]
        if not self.block_names:
            raise ValueError(
                f'hook 을 걸 블록을 찾지 못했습니다. 후보={names}, '
                f'실제 속성={[k for k, v in netG.named_children()]}')

    def forward(self, x, feat=False):
        if not feat:
            return self.netG(x)

        feats = []
        handles = []
        for name in self.block_names:
            m = getattr(self.netG, name)
            handles.append(m.register_forward_hook(
                lambda mod, inp, out: feats.append(out)))   # detach 하지 않는다
        try:
            out = self.netG(x)
        finally:
            for h in handles:
                h.remove()

        # 원본이 (output, features) 를 반환하는 경우 output 만 취한다
        if isinstance(out, tuple):
            out = out[0]
        return out, feats

    # 편의 위임 — student.parameters(), student.train() 등이 그대로 동작하도록
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.netG, name)
