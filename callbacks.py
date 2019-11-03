from keras.callbacks import Callback
from utils import visualize_segmentation_result

class VisualizeResults(Callback):

    def __init__(self, images, masks, model, n_classes, output_path):

        self.images = images
        self.masks = masks
        self.model = model
        self.n_classes = n_classes
        self.output_path = output_path
    
    def on_epoch_end(self, epoch, logs=None):

        preds = self.model.predict(self.images)
        visualize_segmentation_result(self.images, self.masks, preds, figsize=4, nm_img_to_plot=100, 
                                        n_classes=self.n_classes, ouput_path=self.output_path, epoch=epoch)
        
