from keras.applications.vgg16 import VGG16
from matplotlib import pyplot as plt

from vis.utils import utils
from vis.visualization import visualize_activation

model = VGG16(
    weights='./data/vgg16_weights_tf_dim_ordering_tf_kernels.h5', include_top=True)
print(model.summary())

layer_name = 'predictions'
layer_idx = [
    idx for idx, layer in enumerate(model.layers)
    if layer.name == layer_name
][0]
vis_image = []
for idx in [20, 20, 20]:
    img = visualize_activation(
        model, layer_idx, filter_indices=idx, max_iter=500)
    img = utils.draw_text(img, str(idx))
    vis_image.append(img)
stitched = utils.stitch_images(vis_image)
plt.imshow(stitched)
plt.show()

layer_name = 'block3_conv1'
layer_idx = [
    idx for idx, layer in enumerate(model.layers)
    if layer.name == layer_name
][0]
vis_image = []
for idx in [20, 20, 20]:
    img = visualize_activation(
        model, layer_idx, filter_indices=idx, max_iter=500)
    img = utils.draw_text(img, str(idx))
    vis_image.append(img)
stitched = utils.stitch_images(vis_image)
plt.imshow(stitched)
plt.show()
