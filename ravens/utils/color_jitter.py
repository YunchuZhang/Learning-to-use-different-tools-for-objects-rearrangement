import numpy as np 
from PIL import ImageEnhance, Image

transform_type_dict = dict(
	brightness=ImageEnhance.Brightness, contrast=ImageEnhance.Contrast,
	sharpness=ImageEnhance.Sharpness,   color=ImageEnhance.Color
)

class ColorJitter(object):
	def __init__(self, transform_dict):
		self.transforms = [(transform_type_dict[k], transform_dict[k]) for k in transform_dict]
	
	def __call__(self, img):
		out = img 
		rand_num = np.random.uniform(0, 1, len(self.transforms))

		for i, (transformer, alpha) in enumerate(self.transforms):
			r = alpha * (rand_num[i]*2.0 - 1.0) + 1   # r in [1-alpha, 1+alpha)
			out = transformer(out).enhance(r)
		
		return out
def adjust_hue(img, hue_factor):
	"""Adjust hue of an image.
	The image hue is adjusted by converting the image to HSV and
	cyclically shifting the intensities in the hue channel (H).
	The image is then converted back to original image mode.
	`hue_factor` is the amount of shift in H channel and must be in the
	interval `[-0.5, 0.5]`.
	See https://en.wikipedia.org/wiki/Hue for more details on Hue.
	Args:
		img (PIL Image): PIL Image to be adjusted.
		hue_factor (float):  How much to shift the hue channel. Should be in
			[-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
			HSV space in positive and negative direction respectively.
			0 means no shift. Therefore, both -0.5 and 0.5 will give an image
			with complementary colors while 0 gives the original image.
	Returns:
		PIL Image: Hue adjusted image.
	"""
	if not(-0.5 <= hue_factor <= 0.5):
		raise ValueError('hue_factor is not in [-0.5, 0.5].'.format(hue_factor))

	# if not _is_pil_image(img):
	# 	raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

	input_mode = img.mode
	if input_mode in {'L', '1', 'I', 'F'}:
		return img

	h, s, v = img.convert('HSV').split()

	np_h = np.array(h, dtype=np.uint8)
	# uint8 addition take cares of rotation across boundaries
	with np.errstate(over='ignore'):
		np_h += np.uint8(hue_factor * 255)
	h = Image.fromarray(np_h, 'L')

	img = Image.merge('HSV', (h, s, v)).convert(input_mode)
	return img