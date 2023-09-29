class HyperParameter(object):
  def __init__(self):
    self.label_dict    = dict( background = 0,
                               landslide  = 1)

    # Parameter for generator and training
    self.H             = 128    # Height
    self.W             = 128    # Width
    self.C             = 19     # Channel
    self.batch_size    = 20 # 12
    self.start_batch   = 0
    self.learning_rate = 4e-4
    self.is_aug        = True
    self.check_every   = 20
    self.class_num     = 2
    self.epoch_num     = 273
    self.img_mean      = [1111.81236406, 824.63171476, 663.41636217, 445.17289745,
                          645.8582926, 1547.73508126, 1960.44401001, 1941.32229668,
                          674.07572865, 9.04787384, 1113.98338755, 519.90397929,
                          20.29228266, 772.83144788]   # means of 14 bands
    self.img_max       = [3453., 16296., 20932., 14762.,
                          6441.,  6414.,  7239., 16138.,  2392.,
                          194., 6446., 10222.,
                          82.,  3958.]
