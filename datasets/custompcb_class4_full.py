from .custompcb import build_custompcb_variant


def build(image_set, args):
    return build_custompcb_variant(
        image_set,
        args,
        path_attr='custom_pcb_path',
        train_ann_name='custom_train_class4.json',
        val_ann_name='custom_val_class4.json',
    )
