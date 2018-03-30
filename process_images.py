import os, PIL
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import math

def _compose_alpha(img_in, img_layer, opacity):

    comp_alpha = np.minimum(img_in[:, :, 3], img_layer[:, :, 3])*opacity

    new_alpha = img_in[:, :, 3] + (1.0 - img_in[:, :, 3])*comp_alpha
    np.seterr(divide='ignore', invalid='ignore')
    ratio = comp_alpha/new_alpha
    ratio[ratio == np.NAN] = 0.0
    return ratio

def multiply(img_in, img_layer, opacity):

    img_in /= 255.0
    img_layer /= 255.0

    ratio = _compose_alpha(img_in, img_layer, opacity)

    comp = np.clip(img_layer[:, :, :3] * img_in[:, :, :3], 0.0, 1.0)

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp * ratio_rs + img_in[:, :, :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_in[:, :, 3])))  # add alpha channel and replace nans
    return img_out*255.0

def darken_only(img_in, img_layer, opacity):

    img_in /= 255.0
    img_layer /= 255.0

    ratio = _compose_alpha(img_in, img_layer, opacity)

    comp = np.minimum(img_in[:, :, :3], img_layer[:, :, :3])

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp*ratio_rs + img_in[:, :, :3] * (1.0-ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_in[:, :, 3])))  # add alpha channel and replace nans
    return img_out*255.0


def run(gen_size, gen_stride = 1, des_stride = 1, mode = 2, make_index = True, mix = .8, aspect = 2.0):

	files = os.listdir(os.getcwd())
	imlist = [filename for filename in files if  filename[-4:] in [".png",".PNG", ".jpg", ".JPG"]]

	des_nums = []

	for im in imlist:
		try:
			des_nums.append(int(im.split('.')[0]))
		except ValueError:
			des_nums.append(None)
			print "could not process image with name:", im

	print "found", len(des_nums), "images"

	chunks = [range(x, x+gen_size, des_stride) for x in range(0, max(des_nums), gen_size)]
	chunks = [c for i,c in enumerate(chunks) if i % gen_stride == 0]

	im_sets = []

	for chunk in chunks:
		im_sets.append([])
		for c in chunk:
			try:
				indx = des_nums.index(c)
			except ValueError:
				# print "could not find image", c
				continue
			im_sets[-1].append(imlist[indx])

	target_dir = "composites"

	if not os.path.exists(target_dir):
	    os.makedirs(target_dir)

	final_images = []

	for i, im_set in enumerate(im_sets):

		img_id = i * gen_stride

		w,h = Image.open(im_set[0]).size
		num_images = len(im_set)

		txt_height = int(min([w,h]) * 0.05)

		img_out = np.zeros((h,w,4),np.float)
		comp = np.ones((h,w,4),np.float) * 255.0
			
		for im in im_set:
			imarr = np.ones((h,w,4),np.float) * 255.0
			imarr[:,:,:3] = np.array(Image.open(im), dtype=np.float)

			if mode == 1:
				comp = multiply(imarr, comp, mix)
				img_out = img_out + comp / num_images
			elif mode == 2:
				comp = darken_only(imarr, comp, mix)
				img_out = img_out + comp / num_images
			else:
				img_out = img_out + imarr / num_images

		arr = np.ones((h+int(txt_height*1.5),w,3),np.float) * 255.0
		arr[:h,:,:] = img_out[:,:,:3]

		arr = np.array(np.round(arr), dtype=np.uint8)
		out = Image.fromarray(arr, mode="RGB")

		draw = ImageDraw.Draw(out)
		font = ImageFont.truetype("arial.ttf", txt_height)
		draw.text((int(w*0.1), h),str(img_id),font=font,fill=(0,0,0))

		target = target_dir + "/" + str(img_id).zfill(4) + ".png"
		out.save(target)

		final_images.append(out)

		print "saved image to:", target

	if make_index:

		w,h = final_images[0].size

		print w, h

		x_dim = int(math.ceil( aspect * h * len(final_images) / w ) ** 0.5)
		y_dim = int(math.ceil(len(final_images) / float(x_dim)))

		print "making index with dimensions:", x_dim, "x", y_dim

		img_out = np.ones((h*y_dim,w*x_dim,3),np.float) * 255.0

		count = 0

		for y in range(y_dim):
			for x in range(x_dim):
				try:
					imarr = np.array(final_images[count], dtype=np.float)
					img_out[y*h:(y+1)*h,x*w:(x+1)*w,:] = imarr
				except IndexError:
					break
				count += 1

		arr = np.array(np.round(img_out), dtype=np.uint8)
		out = Image.fromarray(arr, mode="RGB")

		target = target_dir + "/index.png"
		out.save(target)

		print "saved image to:", target

if __name__ == "__main__":

	mode = 2
	gen_size = 200
	gen_stride = 5
	des_stride = 1
	make_index = True
	mix = 0.8
	aspect = 2.0

	run(gen_size, gen_stride, des_stride, mode, make_index, mix, aspect)

