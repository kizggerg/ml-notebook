def preprocess(plt, images, target):
  images_and_labels = list(zip(images, target))
  for index, (image, label) in enumerate(images_and_labels[:4]):
      plt.subplot(2, 4, index + 1)
      plt.axis('off')
      plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
      plt.title('Training: %i' % label)

  n_samples = len(images)
  data = images.reshape((n_samples, -1))

  return images, target, n_samples, data
