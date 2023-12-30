template_dict = {
    'aid': [
        lambda c: f'a satellite photo of {c}.',
        lambda c: f'a satellite image of {c}',
    ],
    'eurosat': [
        lambda c: f'a satellite photo of {c}.',
        lambda c: f'a satellite image of {c}',
    ],
    'millionaid': [
        lambda c: f'a satellite photo of {c}.',
        lambda c: f'a satellite image of {c}',
    ],
    'fmow': [
        lambda c: f'a satellite photo of {c}.',
        lambda c: f'a satellite image of {c}',
    ],
    'patternnet': [
        lambda c: f'a satellite photo of {c}.',
        lambda c: f'a satellite image of {c}',
    ],
    'nwpu': [
        lambda c: f'a satellite photo of {c}.',
        lambda c: f'a satellite image of {c}',
    ],
    'SkyScript_cls': [
        lambda c: f'a satellite photo of {c}.',
        lambda c: f'a satellite image of {c}',
    ],
    'rsicb256': [
        lambda c: f'a satellite photo of {c}.',
        lambda c: f'a satellite image of {c}',
    ],
    # fine-grained classification
    'roof_shape': [
        lambda c: f'a satellite photo of building, {c}.',
        lambda c: f'a satellite image of building, {c}',
    ],
    'smoothness': [
        lambda c: f'a satellite photo of road, {c}.',
        lambda c: f'a satellite image of road, {c}',
    ],
    'surface': [
        lambda c: f'a satellite photo of road, {c}.',
        lambda c: f'a satellite image of road, {c}',
    ],
}