from .custompcb import build_custompcb_variant


def build(image_set, args):
    return build_custompcb_variant(
        image_set,
        args,
        path_attr='custom_pcb_gerber_path',
        train_ann_name='custom_train_gerber.json',
        val_ann_name='custom_val_gerber.json',
    )
