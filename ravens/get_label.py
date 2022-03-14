'''
get aruco_pose and finger tip pose in images 
transform to transporter format
'''
import click
import time
import cv2
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from transforms3d import euler
import pickle
import shutil
import sys
sys.path.append("/home/yunchuz/git/detectron2/")

def load_and_seg_spoon(path1,path2,arucoDict,arucoParams,matrix_coefficients,distortion_coefficients):
	# Create a VideoCapture object and read from input file
	# If the input is the camera, pass 0 instead of the video file name
	if not os.path.exists(path1[:-13]+'/keypoints'):
		os.mkdir(path1[:-13]+'/keypoints')
	else:
		shutil.rmtree(path1[:-13]+'/keypoints')
		os.mkdir(path1[:-13]+'/keypoints')
	if not os.path.exists(path1[:-13]+'/rgb'):
		os.mkdir(path1[:-13]+'/rgb')
	else:
		shutil.rmtree(path1[:-13]+'/rgb')
		os.mkdir(path1[:-13]+'/rgb')
	if not os.path.exists(path1[:-13]+'/depth'):
		os.mkdir(path1[:-13]+'/depth')
	else: 
		shutil.rmtree(path1[:-13]+'/depth')
		os.mkdir(path1[:-13]+'/depth')
	# rgb
	cap1 = cv2.VideoCapture(path1)
	#depth
	cap2 = cv2.VideoCapture(path2)
	cap2.read()
	# Check if camera opened successfully
	if (cap1.isOpened()== False): 
		print("Error opening video stream or file")

	i = 0
	cnt = 0
	save_data = {}
	imgs,trans,ori,angle,kp = [],[],[],[],[]
	# Read until video is completed
	w, h = 720, 1280
	_,frame1 = cap1.read()
	crop = frame1[10:80,320:450,:]
	b,g,r = np.mean(crop[:,:,0]),np.mean(crop[:,:,1]),np.mean(crop[:,:,2])
	last_max = np.argmax([b,g,r])
	while(cap1.isOpened()):
		# Capture frame-by-frame

		ret1, frame1 = cap1.read()
		cnt +=1
		ret2, depth = cap2.read()

	
		if ret1 == True :
			crop = frame1[10:80,320:450,:]
			b,g,r = np.mean(crop[:,:,0]),np.mean(crop[:,:,1]),np.mean(crop[:,:,2])
			current_max = np.argmax([b,g,r])
			if current_max != last_max and current_max == 1:
				print(b,g,r)
				print("to green")
				cv2.imwrite(path1[:-13]+'/rgb/'+"img_{}.png".format(i),frame1[360-240:360+240,640-350:640+290,:])
				# cp depth

				depth_path,choose_path = path1[:-13]+'/'+path2[-10:-4]+"/{}".format(cnt),path1[:-13]+'/depth/'+"img_{}".format(i)
				print(depth_path,choose_path )
				cmd = "cp -r {}.npy {}.npy".format(depth_path,choose_path)
				os.system(cmd)
				# generate kp
				if i % 2 != 0:
					kp.append((0,0))
					# trans.append([0,0,0])
					# ori.append([0,0,0,0])
					# angle.append([0,0,0])
					# x y z w
				else:
					
					corners, ids, rejected_img_points = cv2.aruco.detectMarkers(frame1, arucoDict,
															parameters=arucoParams,)

					centerx = int((corners[0][0][0][0] + corners[0][0][2][0]) / 2)
					centery = int((corners[0][0][0][1] + corners[0][0][2][1]) / 2)
					direction = [corners[0][0][0][0] - corners[0][0][3][0], corners[0][0][0][1] - corners[0][0][3][1]]
					direction = direction/np.linalg.norm(direction)
					topx = int(centerx+direction[0]*110)
					topy = int(centery+direction[1]*110)
					crop_x, crop_y = 290, 120
					centerx -= crop_x
					centery -= crop_y
					topx -= crop_x
					topy -= crop_y
					cv2.circle(frame1[360-240:360+240,640-350:640+290,:],(centerx,centery),4,(0,0,255),-1)
					cv2.circle(frame1[360-240:360+240,640-350:640+290,:],(topx,topy),4,(0,255,0),-1)
					cv2.imwrite(path1[:-13]+'/keypoints/'+"img_{}.png".format(i),frame1[360-240:360+240,640-350:640+290,:])
					# cv2.imshow('kps',frame1[360-240:360+240,640-350:640+290,:])
					kp.append((topx,topy))

				time.sleep(1)
				i+=1

			if current_max != last_max and current_max == 2:
				print(b,g,r)
				print("to red")
				cv2.imwrite(path1[:-13]+'/rgb/'+"img_{}.png".format(i),frame1[360-240:360+240,640-350:640+290,:])
				# cp depth
				depth_path,choose_path = path1[:-13]+'/'+path2[-10:-4]+"/{}".format(cnt),path1[:-13]+'/depth/'+"img_{}".format(i)
				print(depth_path,choose_path )
				cmd = "cp -r {}.npy {}.npy".format(depth_path,choose_path)
				os.system(cmd)

				# generate kp
				if i % 2 != 0:
					kp.append((0,0))
					# trans.append([0,0,0])
					# ori.append([0,0,0,0])
					# angle.append([0,0,0])
					# x y z w
				else:
					
					corners, ids, rejected_img_points = cv2.aruco.detectMarkers(frame1, arucoDict,
															parameters=arucoParams,)

					centerx = int((corners[0][0][0][0] + corners[0][0][2][0]) / 2)
					centery = int((corners[0][0][0][1] + corners[0][0][2][1]) / 2)
					direction = [corners[0][0][0][0] - corners[0][0][3][0], corners[0][0][0][1] - corners[0][0][3][1]]
					direction = direction/np.linalg.norm(direction)
					topx = int(centerx+direction[0]*110)
					topy = int(centery+direction[1]*110)
					crop_x, crop_y = 290, 120
					centerx -= crop_x
					centery -= crop_y
					topx -= crop_x
					topy -= crop_y
					cv2.circle(frame1[360-240:360+240,640-350:640+290,:],(centerx,centery),4,(0,0,255),-1)
					cv2.circle(frame1[360-240:360+240,640-350:640+290,:],(topx,topy),4,(0,255,0),-1)
					cv2.imwrite(path1[:-13]+'/keypoints/'+"img_{}.png".format(i),frame1[360-240:360+240,640-350:640+290,:])
					# cv2.imshow('kps',frame1[360-240:360+240,640-350:640+290,:])
					kp.append((topx,topy))

				time.sleep(1)
				i+=1

			last_max = current_max

			cv2.imshow('frame',frame1)
			if ret2 == True:
				cv2.imshow('depth',depth)
			cv2.imshow('crop',crop)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		else: 
			break
	# When everything done, release the video capture object
	print(i,cnt)
	# save_data["trans"] = np.asarray(trans)
	# save_data["ori"] = np.asarray(ori)
	# save_data["angle"] = np.asarray(angle)

	save_data["kp"] = np.asarray(kp)
	print(save_data)
	np.save(path1[:-13]+"/marker_rgb"+path1[-5:-4],save_data)

	save_data = {}
	cap1.release()
	cap2.release()

def load_and_seg_bot(path1,path2,arucoDict,arucoParams,matrix_coefficients,distortion_coefficients):
	# Create a VideoCapture object and read from input file
	# If the input is the camera, pass 0 instead of the video file name
	if not os.path.exists(path1[:-13]+'/keypoints'):
		os.mkdir(path1[:-13]+'/keypoints')
	else:
		shutil.rmtree(path1[:-13]+'/keypoints')
		os.mkdir(path1[:-13]+'/keypoints')
	if not os.path.exists(path1[:-13]+'/rgb'):
		os.mkdir(path1[:-13]+'/rgb')
	else:
		shutil.rmtree(path1[:-13]+'/rgb')
		os.mkdir(path1[:-13]+'/rgb')
	if not os.path.exists(path1[:-13]+'/depth'):
		os.mkdir(path1[:-13]+'/depth')
	else: 
		shutil.rmtree(path1[:-13]+'/depth')
		os.mkdir(path1[:-13]+'/depth')
	# rgb
	cap1 = cv2.VideoCapture(path1)
	#depth
	cap2 = cv2.VideoCapture(path2)
	cap2.read()
	# Check if camera opened successfully
	if (cap1.isOpened()== False): 
		print("Error opening video stream or file")

	i = 0
	cnt = 0
	save_data = {}
	imgs,trans,ori,angle,kp = [],[],[],[],[]
	# Read until video is completed
	w, h = 720, 1280
	_,frame1 = cap1.read()
	crop = frame1[10:80,320:450,:]
	b,g,r = np.mean(crop[:,:,0]),np.mean(crop[:,:,1]),np.mean(crop[:,:,2])
	last_max = np.argmax([b,g,r])
	while(cap1.isOpened()):
		# Capture frame-by-frame

		ret1, frame1 = cap1.read()
		cnt +=1
		if cnt < 630:
			continue
		if i > 172:
			break
		ret2, depth = cap2.read()

	
		if ret1 == True :
			crop = frame1[10:80,320:450,:]
			b,g,r = np.mean(crop[:,:,0]),np.mean(crop[:,:,1]),np.mean(crop[:,:,2])
			current_max = np.argmax([b,g,r])
			if current_max != last_max and current_max == 1:
				print(b,g,r)
				print("to green")
				cv2.imwrite(path1[:-13]+'/rgb/'+"img_{}.png".format(i),frame1[360-240:360+240,640-350:640+290,:])
				# cp depth

				depth_path,choose_path = path1[:-13]+'/'+path2[-10:-4]+"/{}".format(cnt),path1[:-13]+'/depth/'+"img_{}".format(i)
				print(depth_path,choose_path )
				cmd = "cp -r {}.npy {}.npy".format(depth_path,choose_path)
				os.system(cmd)
				# generate kp
				if i % 2 != 0:
					kp.append((0,0,0,0))
					trans.append([0,0,0])
					ori.append([0,0,0,0])
					angle.append([0,0,0])
					# x y z w
				else:
					tmp = path1[:-13]+'/rgb/'
					cmd = "python /home/yunchuz/git/detectron2/demo/demo.py \
					--config-file /home/yunchuz/git/detectron2/configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x_grabber.yaml \
					--input {}img_{}.png \
					--output {}/keypoints \
					--opts MODEL.WEIGHTS /home/yunchuz/git/detectron2/Tool_models/grabber_8.9/model_final.pth > {}/keypoints/keypoints.txt ".format(tmp,i,path1[:-13],path1[:-13])
					os.system(cmd)
					f = open('{}/keypoints/keypoints.txt'.format(path1[:-13]))

					lines = f.readlines()

					assert lines[2][:10]=='keypoint 0'
					assert lines[3][:10]=='keypoint 1'

					kp.append((float(lines[2][13:-2].split(',')[0]),\
					float(lines[2][13:-2].split(',')[1]),\
					float(lines[3][13:-2].split(',')[0]),\
					float(lines[3][13:-2].split(',')[1])))
					

					corners, ids, rejected_img_points = cv2.aruco.detectMarkers(frame1[360-240:360+240,640-350:640+290,:], arucoDict,
															parameters=arucoParams,)

					if np.all(ids is not None) and len(ids)==1 and ids == 39:  # If there are markers found by detector
						# Estimate pose of each marker and return the values rvec and tvec---different from camera coefficients
						rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[0], 0.053, matrix_coefficients,
																						   distortion_coefficients)

						# cv2.Rodrigues(rvec)[0]
						q1 = R.from_matrix(cv2.Rodrigues(rvec)[0]).as_quat()

						imgs.append(i)
						trans.append(tvec[0][0])
						ori.append(q1)
						print(tvec[0][0],q1)
						angle.append(R.from_matrix(cv2.Rodrigues(rvec)[0]).as_euler('zyx', degrees=True))
						# x y z w
						# (rvec - tvec).any()  # get rid of that nasty numpy value array error
						cv2.aruco.drawDetectedMarkers(frame1, corners)  # Draw A square around the markers
						cv2.aruco.drawAxis(frame1, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.1)  # Draw Axis

				time.sleep(1)
				i+=1

			if current_max != last_max and current_max == 2:
				print(b,g,r)
				print("to red")
				cv2.imwrite(path1[:-13]+'/rgb/'+"img_{}.png".format(i),frame1[360-240:360+240,640-350:640+290,:])
				# cp depth
				depth_path,choose_path = path1[:-13]+'/'+path2[-10:-4]+"/{}".format(cnt),path1[:-13]+'/depth/'+"img_{}".format(i)
				print(depth_path,choose_path )
				cmd = "cp -r {}.npy {}.npy".format(depth_path,choose_path)
				os.system(cmd)
				# generate kp
				if i % 2 != 0:
					kp.append((0,0,0,0))
					trans.append([0,0,0])
					ori.append([0,0,0,0])
					angle.append([0,0,0])
					# x y z w
				else:
					tmp = path1[:-13]+'/rgb/'
					cmd = "python /home/yunchuz/git/detectron2/demo/demo.py \
					--config-file /home/yunchuz/git/detectron2/configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x_grabber.yaml \
					--input {}img_{}.png \
					--output {}/keypoints \
					--opts MODEL.WEIGHTS /home/yunchuz/git/detectron2/Tool_models/grabber_8.9/model_final.pth > {}/keypoints/keypoints.txt ".format(tmp,i,path1[:-13],path1[:-13])
					os.system(cmd)
					f = open('{}/keypoints/keypoints.txt'.format(path1[:-13]))
					lines = f.readlines()

					assert lines[2][:10]=='keypoint 0'
					assert lines[3][:10]=='keypoint 1'

					kp.append((float(lines[2][13:-2].split(',')[0]),\
					float(lines[2][13:-2].split(',')[1]),\
					float(lines[3][13:-2].split(',')[0]),\
					float(lines[3][13:-2].split(',')[1])))
					

					corners, ids, rejected_img_points = cv2.aruco.detectMarkers(frame1[360-240:360+240,640-350:640+290,:], arucoDict,
															parameters=arucoParams,)

					if np.all(ids is not None) and len(ids)==1 and ids == 39:  # If there are markers found by detector
						# Estimate pose of each marker and return the values rvec and tvec---different from camera coefficients
						rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[0], 0.053, matrix_coefficients,
																						   distortion_coefficients)

						# cv2.Rodrigues(rvec)[0]
						q1 = R.from_matrix(cv2.Rodrigues(rvec)[0]).as_quat()

						imgs.append(i)
						trans.append(tvec[0][0])
						ori.append(q1)
						angle.append(R.from_matrix(cv2.Rodrigues(rvec)[0]).as_euler('zyx', degrees=True))
						# x y z w
						# (rvec - tvec).any()  # get rid of that nasty numpy value array error
						cv2.aruco.drawDetectedMarkers(frame1, corners)  # Draw A square around the markers
						cv2.aruco.drawAxis(frame1, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.1)  # Draw Axis

				time.sleep(1)
				i+=1
			last_max = current_max
			cv2.imshow('frame',frame1)
			if ret2 == True:
				cv2.imshow('depth',depth)
			cv2.imshow('crop',crop)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		else: 
			break
	# When everything done, release the video capture object
	print(i,cnt)
	save_data["trans"] = np.asarray(trans)
	save_data["ori"] = np.asarray(ori)
	# save_data["angle"] = np.asarray(angle)

	save_data["kp"] = np.asarray(kp)
	print(save_data)
	np.save(path1[:-13]+"/marker_rgb"+path1[-5:-4],save_data)

	save_data = {}
	cap1.release()
	cap2.release()


def load_and_seg_shovel(path1,path2,matrix_coefficients,distortion_coefficients):
	# Create a VideoCapture object and read from input file
	# If the input is the camera, pass 0 instead of the video file name
	if not os.path.exists(path1[:-13]+'/keypoints'):
		os.mkdir(path1[:-13]+'/keypoints')
	else:
		shutil.rmtree(path1[:-13]+'/keypoints')
		os.mkdir(path1[:-13]+'/keypoints')

	if not os.path.exists(path1[:-13]+'/rgb'):
		os.mkdir(path1[:-13]+'/rgb')
	else:
		shutil.rmtree(path1[:-13]+'/rgb')
		os.mkdir(path1[:-13]+'/rgb')
	if not os.path.exists(path1[:-13]+'/depth'):
		os.mkdir(path1[:-13]+'/depth')
	else: 
		shutil.rmtree(path1[:-13]+'/depth')
		os.mkdir(path1[:-13]+'/depth')
	# rgb
	cap1 = cv2.VideoCapture(path1)
	#depth
	cap2 = cv2.VideoCapture(path2)
	cap2.read()
	# Check if camera opened successfully
	if (cap1.isOpened()== False): 
		print("Error opening video stream or file")

	i = 0
	cnt = 0
	save_data = {}
	imgs,trans,ori,angle,kp = [],[],[],[],[]
	# Read until video is completed
	w, h = 720, 1280
	_,frame1 = cap1.read()
	crop = frame1[10:80,320:450,:]
	b,g,r = np.mean(crop[:,:,0]),np.mean(crop[:,:,1]),np.mean(crop[:,:,2])
	last_max = np.argmax([b,g,r])
	while(cap1.isOpened()):
		# Capture frame-by-frame
		ret1, frame1 = cap1.read()
		cnt +=1
		ret2, depth = cap2.read()
		# if i > 343:
		# 	break
		if ret1 == True :
			crop = frame1[10:80,320:450,:]
			b,g,r = np.mean(crop[:,:,0]),np.mean(crop[:,:,1]),np.mean(crop[:,:,2])
			current_max = np.argmax([b,g,r])
			if current_max != last_max and current_max == 1:
				print(b,g,r)
				print("to green")
				cv2.imwrite(path1[:-13]+'/rgb/'+"img_{}.png".format(i),frame1)

				# cp depth

				depth_path,choose_path = path1[:-13]+'/'+path2[-10:-4]+"/{}".format(cnt),path1[:-13]+'/depth/'+"img_{}".format(i)
				print(depth_path,choose_path )
				cmd = "cp -r {}.npy {}.npy".format(depth_path,choose_path)
				os.system(cmd)
				# generate kp
				if i % 2 != 0:
					kp.append((0,0,0,0,0,0))

					# x y z w
				else:
					tmp = path1[:-13]+'/rgb/'
					cmd = "python /home/yunchuz/git/detectron2/demo/demo.py \
					--config-file /home/yunchuz/git/detectron2/configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x_shovel.yaml \
					--input {}img_{}.png \
					--output {}/keypoints \
					--opts MODEL.WEIGHTS /home/yunchuz/git/detectron2/Tool_models/shovel_9.1_3x/model_final.pth > {}/keypoints/keypoints.txt ".format(tmp,i,path1[:-13],path1[:-13])
					os.system(cmd)
					f = open('{}/keypoints/keypoints.txt'.format(path1[:-13]))

					lines = f.readlines()

					assert lines[2][:10]=='keypoint 0'
					assert lines[3][:10]=='keypoint 1'
					assert lines[4][:10]=='keypoint 2'

					crop_x, crop_y = 290, 120
					kp.append((float(lines[2][13:-2].split(',')[0])-crop_x,\
					float(lines[2][13:-2].split(',')[1])-crop_y,\
					float(lines[3][13:-2].split(',')[0])-crop_x,\
					float(lines[3][13:-2].split(',')[1])-crop_y,\
					float(lines[4][13:-2].split(',')[0])-crop_x,\
					float(lines[4][13:-2].split(',')[1])-crop_y))
					

					
					# # cv2.Rodrigues(rvec)[0]
					# q1 = R.from_matrix(cv2.Rodrigues(rvec)[0]).as_quat()

					# imgs.append(i)
					# trans.append(tvec[0][0])
					# ori.append(q1)
					# print(tvec[0][0],q1)
					# angle.append(R.from_matrix(cv2.Rodrigues(rvec)[0]).as_euler('zyx', degrees=True))
					# # x y z w
					# # (rvec - tvec).any()  # get rid of that nasty numpy value array error
					# cv2.aruco.drawDetectedMarkers(frame1, corners)  # Draw A square around the markers
					# cv2.aruco.drawAxis(frame1, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.1)  # Draw Axis

				time.sleep(1)
				i+=1

			if current_max != last_max and current_max == 2:
				print(b,g,r)
				print("to red")
				cv2.imwrite(path1[:-13]+'/rgb/'+"img_{}.png".format(i),frame1)

				# cp depth
				depth_path,choose_path = path1[:-13]+'/'+path2[-10:-4]+"/{}".format(cnt),path1[:-13]+'/depth/'+"img_{}".format(i)
				print(depth_path,choose_path )
				cmd = "cp -r {}.npy {}.npy".format(depth_path,choose_path)
				os.system(cmd)
				# generate kp
				if i % 2 != 0:
					kp.append((0,0,0,0,0,0))
					# x y z w
				else:
					tmp = path1[:-13]+'/rgb/'
					cmd = "python /home/yunchuz/git/detectron2/demo/demo.py \
					--config-file /home/yunchuz/git/detectron2/configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x_shovel.yaml \
					--input {}img_{}.png \
					--output {}/keypoints \
					--opts MODEL.WEIGHTS /home/yunchuz/git/detectron2/Tool_models/shovel_9.1_3x/model_final.pth > {}/keypoints/keypoints.txt ".format(tmp,i,path1[:-13],path1[:-13])
					os.system(cmd)
					f = open('{}/keypoints/keypoints.txt'.format(path1[:-13]))
					lines = f.readlines()

					assert lines[2][:10]=='keypoint 0'
					assert lines[3][:10]=='keypoint 1'
					assert lines[4][:10]=='keypoint 2'

					crop_x, crop_y = 290, 120
					kp.append((float(lines[2][13:-2].split(',')[0])-crop_x,\
					float(lines[2][13:-2].split(',')[1])-crop_y,\
					float(lines[3][13:-2].split(',')[0])-crop_x,\
					float(lines[3][13:-2].split(',')[1])-crop_y,\
					float(lines[4][13:-2].split(',')[0])-crop_x,\
					float(lines[4][13:-2].split(',')[1])-crop_y))
					

					
					# # cv2.Rodrigues(rvec)[0]
					# q1 = R.from_matrix(cv2.Rodrigues(rvec)[0]).as_quat()

					# imgs.append(i)
					# trans.append(tvec[0][0])
					# ori.append(q1)
					# angle.append(R.from_matrix(cv2.Rodrigues(rvec)[0]).as_euler('zyx', degrees=True))
					# # x y z w
					# # (rvec - tvec).any()  # get rid of that nasty numpy value array error
					# cv2.aruco.drawDetectedMarkers(frame1, corners)  # Draw A square around the markers
					# cv2.aruco.drawAxis(frame1, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.1)  # Draw Axis

				time.sleep(1)
				i+=1
			last_max = current_max
			cv2.imshow('frame',frame1)
			if ret2 == True:
				cv2.imshow('depth',depth)
			cv2.imshow('crop',crop)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		else: 
			break
	# When everything done, release the video capture object
	print(i,cnt)
	# save_data["trans"] = np.asarray(trans)
	# save_data["ori"] = np.asarray(ori)
	# save_data["angle"] = np.asarray(angle)

	save_data["kp"] = np.asarray(kp)
	print(save_data)
	np.save(path1[:-13]+"/marker_rgb"+path1[-5:-4],save_data)

	save_data = {}
	cap1.release()
	cap2.release()

def xy_filter(x1, y1, x2, y2, x3, y3):
	'''
	based on mid point to decide which one is point1, which is point2 
	then decides the rotation direction
	(x1,y1) (x2,y2)
	'''
	# find the mid line angle with x axis (pixel coord x-->right, y-->down)
	mid_x, mid_y = int((x1+x2)/2), int((y1+y2)/2)
	v0 = [1 - 0,0 - 0]
	v_mid = [mid_x - x3,mid_y - y3]
	uv0 = v0/np.linalg.norm(v0)
	uv_mid = v_mid/np.linalg.norm(v_mid)
	dot = uv0.dot(uv_mid)
	ang = np.arccos(dot)*180/np.pi
	print(ang)
	if np.abs(90-ang) < 6:
		print("nearly vertical here")
		print(ang)
		x1_new, y1_new = x1, y1 
		x2_new, y2_new = x2, y2 
		return x1_new, y1_new, x2_new, y2_new, ang 
	if ang < 8 or ang > 173:
		print("nearly horizental here")
		print(ang)
		x1_new, y1_new = x1, y1 
		x2_new, y2_new = x2, y2 
		return x1_new, y1_new, x2_new, y2_new, ang
	if y3 > mid_y:
		if ang < 90:
			if x1 < x2:
				x1_new, y1_new = x1, y1 
				x2_new, y2_new = x2, y2 
				assert y1_new < y2_new
			else:
				x1_new, y1_new = x2, y2 
				x2_new, y2_new = x1, y1 
				assert y1_new < y2_new
		else:
			if x1 < x2:
				x1_new, y1_new = x1, y1 
				x2_new, y2_new = x2, y2 
				assert y1_new > y2_new
			else:
				x1_new, y1_new = x2, y2 
				x2_new, y2_new = x1, y1 
				assert y1_new > y2_new
	else:
		if ang < 90:
			if x1 > x2:
				x1_new, y1_new = x1, y1 
				x2_new, y2_new = x2, y2
				assert y1_new < y2_new
			else:
				x1_new, y1_new = x2, y2 
				x2_new, y2_new = x1, y1 
				assert y1_new < y2_new
		else:
			if x1 > x2:
				x1_new, y1_new = x1, y1 
				x2_new, y2_new = x2, y2 
				assert y1_new > y2_new
			else:
				x1_new, y1_new = x2, y2 
				x2_new, y2_new = x1, y1 
				assert y1_new > y2_new

	return x1_new, y1_new, x2_new, y2_new, ang 

def totransportor_shovel(path, tid=2, is_mix=False):
	'''
	parse current rgb, depth, kp, array to transporter format
	color: 2*views*480*640*3
	depth: 2* views*480*640
	action: [{pose0:(po,qu),pose1:(po,qu)},None]
	info:
	reward:
	'''
	if os.path.exists(path+'/tran'):
		shutil.rmtree(path+'/tran')
	os.mkdir(path+'/tran')
	os.mkdir(path+'/tran/color')
	os.mkdir(path+'/tran/depth')
	os.mkdir(path+'/tran/action')
	os.mkdir(path+'/tran/info')
	os.mkdir(path+'/tran/reward')
	file = np.load(path+"/marker_rgb1.npy",allow_pickle=True)

	# np.save(path+"/marker_rgb1.npy",file)
	# file.item()['kp'][] = file.item()['kp'][]
	
	num = file.item()['kp'].shape[0] 
	print(num)
	num = (num //4)*4
	for i in range(0,num,4):
		# if i > 151:
		# 	break
		print(i)
		fname = f'{i//4:06d}-{i}.pkl'
		# color
		color = []
		rgb1 = cv2.imread(path+'/rgb/'+"img_{}.png".format(i+1))[360-240:360+240,640-350:640+290,:]
		rgb2 = cv2.imread(path+'/rgb/'+"img_{}.png".format(i+3))[360-240:360+240,640-350:640+290,:]
		color.append(rgb1[:,:,::-1])
		color.append(rgb2[:,:,::-1])
		color = np.expand_dims(np.uint8(color),axis=1)
		with open(path+'/tran/color/'+fname, 'wb') as fp:
			pickle.dump(color, fp)
		# depth
		# now depth with realsense divided by 1000
		depth = []
		depth1 = np.load(path+'/depth/'+"img_{}.npy".format(i+1))[360-240:360+240,640-350:640+290]
		depth2 = np.load(path+'/depth/'+"img_{}.npy".format(i+3))[360-240:360+240,640-350:640+290]
		depth.append(depth1)
		depth.append(depth2)
		depth = np.expand_dims(np.float32(depth),axis=1)
		with open(path+'/tran/depth/'+fname, 'wb') as fp:
			pickle.dump(depth, fp)
		# action
		## use first object pose as reletive 0 0 0 1 ? ?

		action = []
		ac = {}
		# TODO need transporter to predict picking angle if no suction cup
		# pose0
		x1, y1, x2, y2, x3, y3 = file.item()['kp'][i]
		pick_x, pick_y, pick_z = int((x1+x2)/2), int((y1+y2)/2), 0
		pick_z = depth1[pick_y,pick_x]/1000 if depth1.max() > 100 else depth1[pick_y,pick_x]
		# import ipdb;ipdb.set_trace()
		# unproject 
		pick_x, pick_y = (pick_x - camera1[0][2]) * pick_z / camera1[0][0], \
							(pick_y - camera1[1][2]) * pick_z / camera1[1][1]
		x1_new, y1_new, x2_new, y2_new, mid_ang = xy_filter(x1, y1, x2, y2, x3, y3)

		# based on mid angle to judge the angle arrow direction
		if np.abs(90-mid_ang) < 5:
			ang_0 = 0 
		elif mid_ang < 90:
			v1 = [1, 0]
			v2 = [x2_new-x1_new, y2_new-y1_new]
			uv1 = v1/np.linalg.norm(v1)
			uv2 = v2/np.linalg.norm(v2)
			dot = uv1.dot(uv2)
			ang_0 = np.arccos(dot)*180/np.pi
		else:
			v1 = [1, 0]
			v2 = [x1_new-x2_new, y1_new-y2_new]
			uv1 = v1/np.linalg.norm(v1)
			uv2 = v2/np.linalg.norm(v2)
			dot = uv1.dot(uv2)
			ang_0 = np.arccos(dot)*180/np.pi

		ori = R.from_euler('zxy', [0, 0, 0], degrees=True).as_quat()
		quaternion_wxyz = np.array([ori[3], ori[0], ori[1], ori[2]])
		euler_zxy = euler.quat2euler(quaternion_wxyz, axes='szxy')
		assert euler_zxy[0]*180/np.pi == 0



		ac['pose0'] = (np.asarray([pick_x, pick_y, pick_z]),ori)
		# pose1 
		x1, y1, x2, y2, x3, y3 = file.item()['kp'][i+2]
		pick_x, pick_y, pick_z = int((x1+x2)/2), int((y1+y2)/2), 0
		pick_z = depth2[pick_y,pick_x]/1000 if depth2.max() > 100 else depth2[pick_y,pick_x]
		# unproject 
		pick_x, pick_y = (pick_x - camera1[0][2]) * pick_z / camera1[0][0], \
							(pick_y - camera1[1][2]) * pick_z / camera1[1][1]

		x1_new, y1_new, x2_new, y2_new, mid_ang = xy_filter(x1, y1, x2, y2, x3, y3)


		if np.abs(90-mid_ang) < 5:
			ang_1 = 0 
		elif mid_ang < 90:
			v1 = [1, 0]
			v2 = [x2_new-x1_new, y2_new-y1_new]
			uv1 = v1/np.linalg.norm(v1)
			uv2 = v2/np.linalg.norm(v2)
			dot = uv1.dot(uv2)
			ang_1 = np.arccos(dot)*180/np.pi
		else:
			v1 = [1, 0]
			v2 = [x1_new-x2_new, y1_new-y2_new]
			uv1 = v1/np.linalg.norm(v1)
			uv2 = v2/np.linalg.norm(v2)
			dot = uv1.dot(uv2)
			ang_1 = np.arccos(dot)*180/np.pi


		ori = R.from_euler('zxy', [(ang_0 + ang_1) / 2, 0, 0], degrees=True).as_quat()
		quaternion_wxyz = np.array([ori[3], ori[0], ori[1], ori[2]])
		euler_zxy = euler.quat2euler(quaternion_wxyz, axes='szxy')
		assert (euler_zxy[0]*180/np.pi - (ang_0 + ang_1) / 2) < 1e-4
		print((ang_0 + ang_1) / 2)
		ac['pose1'] = (np.asarray([pick_x, pick_y, pick_z]),ori)

		ac['tid'] = tid
		ac['is_mix'] = is_mix
		action.append(ac)
		action.append(None)
		with open(path+'/tran/action/'+fname, 'wb') as fp:
			pickle.dump(action, fp)

		info,reward = [None,(1,1)],[0,1]
		with open(path+'/tran/info/'+fname, 'wb') as fp:
			pickle.dump(info, fp)

		with open(path+'/tran/reward/'+fname, 'wb') as fp:
			pickle.dump(reward, fp)

def totransportor_spoon(path, tid=3, is_mix=False):
	'''
	parse current rgb, depth, kp, array to transporter format
	color: 2*views*480*640*3
	depth: 2* views*480*640
	action: [{pose0:(po,qu),pose1:(po,qu)},None]
	info:
	reward:
	'''
	if os.path.exists(path+'/tran'):
		shutil.rmtree(path+'/tran')
	os.mkdir(path+'/tran')
	os.mkdir(path+'/tran/color')
	os.mkdir(path+'/tran/depth')
	os.mkdir(path+'/tran/action')
	os.mkdir(path+'/tran/info')
	os.mkdir(path+'/tran/reward')
	file = np.load(path+"/marker_rgb1.npy",allow_pickle=True)

	# np.save(path+"/marker_rgb1.npy",file)
	# file.item()['kp'][] = file.item()['kp'][]
	
	num = file.item()['kp'].shape[0] 
	print(num)
	num = (num //4)*4
	for i in range(0,num,4):
		# if i > 151:
		# 	break
		print(i)
		fname = f'{i//4:06d}-{i}.pkl'
		# color
		color = []
		rgb1 = cv2.imread(path+'/rgb/'+"img_{}.png".format(i+1))
		rgb2 = cv2.imread(path+'/rgb/'+"img_{}.png".format(i+3))
		color.append(rgb1[:,:,::-1])
		color.append(rgb2[:,:,::-1])
		color = np.expand_dims(np.uint8(color),axis=1)
		with open(path+'/tran/color/'+fname, 'wb') as fp:
			pickle.dump(color, fp)
		# depth
		# now depth with realsense divided by 1000
		depth = []
		depth1 = np.load(path+'/depth/'+"img_{}.npy".format(i+1))[360-240:360+240,640-350:640+290]
		depth2 = np.load(path+'/depth/'+"img_{}.npy".format(i+3))[360-240:360+240,640-350:640+290]
		depth.append(depth1)
		depth.append(depth2)
		depth = np.expand_dims(np.float32(depth),axis=1)
		with open(path+'/tran/depth/'+fname, 'wb') as fp:
			pickle.dump(depth, fp)
		# action
		## use first object pose as reletive 0 0 0 1 ? ?

		action = []
		ac = {}
		# TODO need transporter to predict picking angle if no suction cup
		# pose0
		pick_x, pick_y = file.item()['kp'][i]
		pick_z = depth1[pick_y,pick_x]/1000 if depth1.max() > 100 else depth1[pick_y,pick_x]
		# import ipdb;ipdb.set_trace()
		# unproject 
		pick_x, pick_y = (pick_x - camera1[0][2]) * pick_z / camera1[0][0], \
							(pick_y - camera1[1][2]) * pick_z / camera1[1][1]



		ori = R.from_euler('zxy', [0, 0, 0], degrees=True).as_quat()
		quaternion_wxyz = np.array([ori[3], ori[0], ori[1], ori[2]])
		euler_zxy = euler.quat2euler(quaternion_wxyz, axes='szxy')
		assert euler_zxy[0]*180/np.pi == 0

		ac['pose0'] = (np.asarray([pick_x, pick_y, pick_z]),ori)
		# pose1 
		pick_x, pick_y = file.item()['kp'][i+2]
		pick_z = depth1[pick_y,pick_x]/1000 if depth1.max() > 100 else depth1[pick_y,pick_x]
		# import ipdb;ipdb.set_trace()
		# unproject 
		pick_x, pick_y = (pick_x - camera1[0][2]) * pick_z / camera1[0][0], \
							(pick_y - camera1[1][2]) * pick_z / camera1[1][1]



		ori = R.from_euler('zxy', [0, 0, 0], degrees=True).as_quat()
		quaternion_wxyz = np.array([ori[3], ori[0], ori[1], ori[2]])
		euler_zxy = euler.quat2euler(quaternion_wxyz, axes='szxy')
		assert euler_zxy[0]*180/np.pi == 0

		ac['pose1'] = (np.asarray([pick_x, pick_y, pick_z]),ori)

		ac['tid'] = tid
		ac['is_mix'] = is_mix
		action.append(ac)
		action.append(None)
		with open(path+'/tran/action/'+fname, 'wb') as fp:
			pickle.dump(action, fp)

		info,reward = [None,(1,1)],[0,1]
		with open(path+'/tran/info/'+fname, 'wb') as fp:
			pickle.dump(info, fp)

		with open(path+'/tran/reward/'+fname, 'wb') as fp:
			pickle.dump(reward, fp)

def totransportor(path,tid =0,is_mix=False):
	'''
	parse current rgb, depth, kp, array to transporter format
	color: 2*views*480*640*3
	depth: 2* views*480*640
	action: [{pose0:(po,qu),pose1:(po,qu)},None]
	info:
	reward:
	tid  0 parallel
		 1 suction
		 2 brush
	'''
	if os.path.exists(path+'/tran'):
		shutil.rmtree(path+'/tran')
	os.mkdir(path+'/tran')
	os.mkdir(path+'/tran/color')
	os.mkdir(path+'/tran/depth')
	os.mkdir(path+'/tran/action')
	os.mkdir(path+'/tran/info')
	os.mkdir(path+'/tran/reward')
	file = np.load(path+"/marker_rgb1.npy",allow_pickle=True)
	num = file.item()['trans'].shape[0] 
	print(num)
	num = (num //4)*4
	for i in range(0,num,4):
		print(i)
		fname = f'{i//4:06d}-{i}.pkl'
		# color
		color = []
		rgb1 = cv2.imread(path+'/rgb/'+"img_{}.png".format(i+1))
		rgb2 = cv2.imread(path+'/rgb/'+"img_{}.png".format(i+3))
		color.append(rgb1[:,:,::-1])
		color.append(rgb2[:,:,::-1])
		color = np.expand_dims(np.uint8(color),axis=1)
		with open(path+'/tran/color/'+fname, 'wb') as fp:
			pickle.dump(color, fp)
		# depth
		# now depth with realsense divided by 1000 
		depth = []
		depth1 = np.load(path+'/depth/'+"img_{}.npy".format(i+1))[360-240:360+240,640-350:640+290]
		depth2 = np.load(path+'/depth/'+"img_{}.npy".format(i+3))[360-240:360+240,640-350:640+290]
		depth.append(depth1)
		depth.append(depth2)
		depth = np.expand_dims(np.float32(depth),axis=1)
		with open(path+'/tran/depth/'+fname, 'wb') as fp:
			pickle.dump(depth, fp)
		# action
		## use first object pose as reletive 0 0 0 1 ? ?

		action = []
		ac = {}
		# TODO need transporter to predict picking angle if no suction cup
		# pose0
		x1, y1, x2, y2 = file.item()['kp'][i]
		pick_x, pick_y, pick_z = int((x1+x2)/2), int((y1+y2)/2), 0
		pick_z = depth1[pick_y,pick_x]/1000 if depth1.max() > 100 else depth1[pick_y,pick_x]
		# import ipdb;ipdb.set_trace()
		# unproject 
		pick_x, pick_y = (pick_x - camera1[0][2]) * pick_z / camera1[0][0], \
							(pick_y - camera1[1][2]) * pick_z / camera1[1][1]
		ac['pose0'] = (np.asarray([pick_x, pick_y, pick_z]),file.item()['ori'][i])
		# pose1 
		x1, y1, x2, y2 = file.item()['kp'][i+2]
		pick_x, pick_y, pick_z = int((x1+x2)/2), int((y1+y2)/2), 0
		pick_z = depth2[pick_y,pick_x]/1000 if depth2.max() > 100 else depth2[pick_y,pick_x]
		# unproject 
		pick_x, pick_y = (pick_x - camera1[0][2]) * pick_z / camera1[0][0], \
							(pick_y - camera1[1][2]) * pick_z / camera1[1][1]
		ac['pose1'] = (np.asarray([pick_x, pick_y, pick_z]),file.item()['ori'][i+2])
		# print("0 --- grasp") 
		# print("1 --- suc")
		# value = input("Please enter current tid :\n")

		# value = int(value)
		# ac['tid'] = value

		# ac['tid'] = tid
		ac['is_mix'] = is_mix
		action.append(ac)
		action.append(None)
		with open(path+'/tran/action/'+fname, 'wb') as fp:
			pickle.dump(action, fp)

		info,reward = [None,(1,1)],[0,1]
		with open(path+'/tran/info/'+fname, 'wb') as fp:
			pickle.dump(info, fp)

		with open(path+'/tran/reward/'+fname, 'wb') as fp:
			pickle.dump(reward, fp)




if __name__ == '__main__':


		# tid  0 parallel
		#  1 suction
		#  2 brush

	# arucoDict and create parameters
	arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
	arucoParams = cv2.aruco.DetectorParameters_create()

	camera1 = np.asarray([610.6864013671875, 0.0, 638.60693359375-290, 0.0, 610.6304931640625, 366.0154724121094-120, 0.0, 0.0, 1.0]).reshape((3,3))
	camera2 = np.asarray([612.0875244140625, 0.0, 322.45697021484375, 0.0, 612.2713623046875, 244.1866455078125, 0.0, 0.0, 1.0]).reshape((3,3))
	camera3 = np.asarray([606.0482177734375, 0.0, 324.88055419921875, 0.0, 605.8035888671875, 239.1813507080078, 0.0, 0.0, 1.0]).reshape((3,3))
	camera4 = np.asarray([615.3900146484375, 0.0, 326.35467529296875, 0.0, 615.323974609375, 240.33250427246094, 0.0, 0.0, 1.0]).reshape((3,3))
	camera5 = np.asarray([605.04345703125, 0.0, 326.4844055175781, 0.0, 605.3517456054688, 242.170166015625, 0.0, 0.0, 1.0]).reshape((3,3))
	camera6 = np.asarray([605.6102905273438, 0.0, 326.23828125, 0.0, 605.5659790039062, 240.68490600585938, 0.0, 0.0, 1.0]).reshape((3,3))
	
	path1 = "/home/yunchuz/gas/cut1/demo1/cut_rgb1.avi"
	path2 = "/home/yunchuz/gas/cut1/demo1/cut_depth1.avi"
	distortion_coefficients = np.asarray([0.0,0,0,0,0])
	# load_and_seg_bot(path1,path2,arucoDict,arucoParams,camera1,distortion_coefficients)
	# load_and_seg_shovel(path1,path2,camera1,distortion_coefficients)
	# load_and_seg_spoon(path1,path2,arucoDict,arucoParams,camera1,distortion_coefficients)
	base_path = "/home/yunchuz/gas/cut1/demo1"
	# base_path = "/home/yunchuz/gas/sweep/demo24"
	import ipdb;ipdb.set_trace()
	# totransportor_shovel(base_path,tid=2)
	# totransportor(base_path,tid=1,is_mix=False)
	# totransportor_spoon(base_path,tid=3,is_mix=False)


	import ipdb;ipdb.set_trace()
	base_folder = "d_train"
	f_list = ["demo13","demo14","demo15","demo16","demo17","demo18","demo19","demo20","demo21","demo22","demo24"]
	f_list = ["demo0","demo2","demo10","demo11","demo12","demo13","demo14","demo15","demo16"]
	f_list = ["demo0","demo1","demo2","demo3","demo4","demo5","demo6","demo7","demo8","demo9", "demo10","demo21","demo22","demo24"]
	# f_list = ["demo3","demo4","demo5"]
	f_list = ["demo1"]

	for add_folder in f_list:
		add_num = len(os.listdir("/home/yunchuz/gas/spoon/{}/tran/color".format(base_folder)))
		import os 
		files = os.listdir("/home/yunchuz/gas/spoon/{}/tran/color".format(add_folder))
		files = sorted(files)
		cnt = len(files)
		for i in range(cnt):
			file = files[i]
			num = i
		# for file in files:
		# 	num = int(file.split('-')[0])
			changename = "{:06d}-{}.pkl".format(num+add_num, 4*(num+add_num))
			for subname in ['color', 'depth', 'action', 'info', 'reward']:
				cmd = "cp -r /home/yunchuz/gas/spoon/{}/tran/{}/{} /home/yunchuz/gas/spoon/{}/tran/{}/{}".format(add_folder,subname,file,base_folder,subname,changename)
				os.system(cmd)
	
	# import ipdb;ipdb.set_trace()
	# path = "/home/yunchuz/gas/mix/demo10/tran/action"
	# files = os.listdir(path)
	# files = sorted(files)
	# cnt = len(files)
	# # for file in files:

	# # 	action = pickle.load( open( path+'/'+file, "rb" ) )
	# # 	action[0]['is_mix'] = True
	# # 	pickle.dump( action, open( path+'/'+file, "wb" ) )
	# import ipdb;ipdb.set_trace()
	# for file in files:

	# 	action = pickle.load( open( path+'/'+file, "rb" ) )
	# 	print(action[0]['is_mix']) 

	# # path4 = "/Users/yunchuz/Downloads/frankapy/cut_rgb4.avi"
	# # distortion_coefficients = np.asarray([0.0,0,0,0,0])
	# # loadVidrealtime(path4,arucoDict,arucoParams,camera4,distortion_coefficients)


