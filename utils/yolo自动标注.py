from ultralytics.data.annotator import auto_annotate

auto_annotate(data=r"F:\work\dataset\rebar2D\train\TEMP", det_model=r"F:\work\python\clone\2d\ultralnew\ultralytics\best1.pt", sam_model=r'F:\work\python\clone\utils\mask\sam_b.pt')