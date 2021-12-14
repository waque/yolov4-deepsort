import numpy as np
import cv2
import os
import pprint
import pandas as pd

class Point():
	def __init__(self, x, y):
		self.x = x
		self.y = y

class NamedQuadri:
	def __init__(self, points, label, person=False):
		self.points = points
		self.label = label
		self.person = person


	def draw(self, cv_label, frame, intersect=False):
		#cv2.rectangle(frame, self.points[0], self.points[2], (0,0,255), 10)
		"""for index, item in enumerate(self.points): 
			if index == len(self.points) - 1:
				cv2.line(frame, item, self.points[0], [0, 255, 0], 2) 
				break
			cv2.line(frame, item, self.points[index + 1], [0, 255, 0], 2) """
		if not self.person:
			color = (255,0,0)
		else:
			color = (0,0,255) if not intersect else (0,255,0)
		cv2.drawContours(frame, [np.array(self.points)], 0, color, 3)

		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(frame, self.label , self.points[1], font, 1, color, 2, cv2.LINE_AA)

	def point_inside(self, point):
		A = []
		B = []
		C = []  
		polygon = self.points
		point = Point(x=point[0], y=point[1])
		for i in range(len(polygon)):
			p1 = Point(polygon[i][0], polygon[i][1])
			p2 = polygon[(i + 1) % len(polygon)]
			p2 = Point(p2[0], p2[1])
			
			# calculate A, B and C
			a = -(p2.y - p1.y)
			b = p2.x - p1.x
			c = -(a * p1.x + b * p1.y)

			A.append(a)
			B.append(b)
			C.append(c)

		D = []
		for i in range(len(A)):
			d = A[i] * point.x + B[i] * point.y + C[i]
			D.append(d)

		t1 = all(d >= 0 for d in D)
		t2 = all(d <= 0 for d in D)
		return t1 or t2
	
	def intersects(self, otherQuadri):
		cond1 = False
		for p in self.points:
			if otherQuadri.point_inside(p):
				cond1 = True

		cond2 = False
		for p in otherQuadri.points:
			if self.point_inside(p):
				cond2 = True
		return cond1 or cond2

	def __str__(self):
		return f"{self.label}:{self.points}"



drawing = False
finished_drawing = False
quadrilaters: NamedQuadri = []
points = []
curr_mouse_pos = None

def draw_rect_callback(event, x, y, flags, params):
	global points, drawing, finished_drawing, curr_mouse_pos
	if event == cv2.EVENT_LBUTTONDOWN:
		curr_mouse_pos = (x, y)
		drawing = True
		print(f"appending {(x,y)}")
		points.append((x, y))
		if len(points) == 4:
			drawing = False
			finished_drawing = True
			new_quadri = NamedQuadri(points, 'loja-' + str(len(quadrilaters)) )
			quadrilaters.append(new_quadri)
			points = []
	elif event == cv2.EVENT_MOUSEMOVE:
		curr_mouse_pos = (x, y)

def print_quadrilaters_info():
	print("----------------------")
	for q in quadrilaters:
		print(q)

	for q1 in quadrilaters:
		for q2 in quadrilaters:
			if q1.label == q2.label:
				continue
			if q1.intersects(q2):
				print(f"INTERSECTION: {q1.label}/{q2.label}")
	print("----------------------")

def rect_painter(vid_path):
	if vid_path is None:
		vid_path = '/Users/miguelreis/Documents/work/other/okli/yolov4-deepsort/outputs/demo.avi'
	img = vid_path
	cv2.namedWindow('video')
	cv2.setMouseCallback('video', draw_rect_callback)

	cap = cv2.VideoCapture(img)
	global points, drawing
	pause=False
	while(cap.isOpened()):

		ret, frame = cap.read()
		if ret == True:
			
			#print current points
			for index, item in enumerate(points): 
				if index == len(points) - 1:
					break
				cv2.line(frame, item, points[index + 1], [255, 255, 0], 3) 
			#print current line
			if drawing and points:
				cv2.line(frame, points[-1], curr_mouse_pos, [0, 0, 255], 3) 

			#print already drawn quadrilaters
			for q in quadrilaters:
				q.draw('video', frame)

			cv2.imshow('video', frame)
			# & 0xFF is required for a 64-bit system
			k = cv2.waitKey(20) & 0xFF
			if k == 27:
				break
			elif k == ord('q'):
				#ouput quadrilaters
				print_quadrilaters_info()
				break
			elif k == ord('p'):
				#ouput quadrilaters
				print_quadrilaters_info()
				cv2.waitKey(-1) #wait until any key is pressed
		else:
			#ouput quadrilaters
			print_quadrilaters_info()
			break
	pprint.pprint(points)
	return quadrilaters

def play_calc_intersects(vid_path, df, zones):
	if vid_path is None:
		vid_path = '/Users/miguelreis/Documents/work/other/okli/yolov4-deepsort/outputs/demo.avi'
	img = vid_path
	cv2.namedWindow('video')
	cv2.setMouseCallback('video', draw_rect_callback)

	cap = cv2.VideoCapture(img)
	pause=False
	curr_frame=0
	while(cap.isOpened()):
		curr_frame +=1

		ret, frame = cap.read()
		if ret == True:
			#print already drawn zones
			for q in zones:
				q.draw('video', frame)
			
			#print yolo people points
			draw_persons_q = df[df.FRAME_ID == curr_frame]
			for _, r in draw_persons_q.iterrows():
				#check first if match with any zone
				p1 = (r.xmin,r.ymin)
				p2 = (r.xmax, r.ymin)
				p3 = (r.xmin, r.ymax)
				p4 = (r.xmax,r.ymax)
				print([p1,p2,p3,p4])
				bbox_quadri= NamedQuadri([p1,p2,p4,p3], "pessoa", person=True)

				person_intersects_any_zone=False
				for q in zones:
					if q.intersects(bbox_quadri):
						person_intersects_any_zone = True
						break
				bbox_quadri.draw('video', frame, intersect=person_intersects_any_zone)

			cv2.imshow('video', frame)
			# & 0xFF is required for a 64-bit system
			k = cv2.waitKey(20) & 0xFF
			if k == 27:
				break
			elif k == ord('q'):
				#ouput quadrilaters
				break
			elif k == ord('p'):
				#ouput quadrilaters
				cv2.waitKey(-1) #wait until any key is pressed
		else:
			#ouput quadrilaters
			print_quadrilaters_info()
			break
	return quadrilaters

def heatmap(frame_number=2):
	img = '/Users/miguelreis/Documents/work/other/okli/yolov4-deepsort/data/video/colombo_example2.mp4'
	cv2.namedWindow('video')
	cv2.setMouseCallback('video', draw_rect)

	points_df = pd.read_csv("/Users/miguelreis/Documents/work/other/okli/yolov4-deepsort/data/out-csv/colombo_example2.csv")

	cap = cv2.VideoCapture(img)

	pause=False
	ret, frame = cap.read()
	i=1
	while i != frame_number:
		ret, frame = cap.read()
		i+=1	

	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	if ret == True:

		#draw points of people
		ps = points_df[points_df.FRAME_ID == frame_number]
		for index, row in ps.iterrows():
			cv2.circle(frame, (int(row.bbox_xcenter), int(row.bbox_ycenter)), 5, (255,0,0), thickness=-1)

		cv2.imshow('video', frame)
		cv2.waitKey()
		import numpy.random as random 
		heatmap_data = random.rand(8,9) 
        #map_img = exposure.rescale_intensity(cam, out_range=(0, 255))
        #map_img = np.uint8(map_img)
		heatmap_img = cv2.applyColorMap(heatmap_data, cv2.COLORMAP_JET)


		#merge map and fram
		fin = cv2.addWeighted(heatmap_img, 0.5, frame, 0.5, 0)

		cv2.imshow('video', fin)
		cv2.waitKey()

first_time=True
import copy
def motion_heatmap(img):
	cv2.namedWindow('cenas')
	capture = cv2.VideoCapture(img)
	length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
	width = capture. get(cv2. CAP_PROP_FRAME_WIDTH )
	height = capture. get(cv2. CAP_PROP_FRAME_HEIGHT )
	print(width, height)
	global first_time
	first_time= True

	def onChangeStart(trackbarValue):
		capture.set(cv2.CAP_PROP_POS_FRAMES,trackbarValue)
		err,img = capture.read()
		cv2.imshow("cenas", img)
		global first_time
		first_time=True
		pass
	def onChange(trackbarValue):
		capture.set(cv2.CAP_PROP_POS_FRAMES,trackbarValue)
		err,img = capture.read()
		cv2.imshow("cenas", img)
		pass

	cv2.createTrackbar( 'start', 'cenas', 0, length, onChangeStart )
	cv2.createTrackbar( 'end'  , 'cenas', 100, length, onChange )
	background_subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()
	#trailing_frames=5
	i=1
	fin = None
	while capture.isOpened():
		i+=1
		ret, frame = capture.read()
		if ret == True:
			if first_time:
				print("FIRSTTIME")
				first_time= False
				first_frame = copy.deepcopy(frame)
				height, width = frame.shape[:2]
				accum_image = np.zeros((height, width), np.uint8)
			filter = background_subtractor.apply(frame)  # remove the background
			threshold = 10
			maxValue = 2
			ret, th1 = cv2.threshold(filter, threshold, maxValue, cv2.THRESH_BINARY)
			accum_image = cv2.add(accum_image, th1)
			color_image_vid = cv2.applyColorMap(accum_image, cv2.COLORMAP_HOT)
			fin = cv2.addWeighted(color_image_vid, 0.4, frame, 0.6, 0)
			cv2.imshow('cenas', fin)
			k = cv2.waitKey(10) & 0xFF
			if k == ord('q'):
				break
		else:
			print("HHHHHEHHERER")
			break

"""def _process_df(df):
  def bbox_to_center(bbox):
    return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
  df['bbox_center'] = df['BBox Coords (xmin, ymin, xmax, ymax)'].apply(bbox_to_center)
  return db"""

def calc_bbox_intersect(df, quadrilaters):
	print(df, quadrilaters)
	for q in quadrilaters:
		df[q.label] = False

	for index, r in df.iterrows():
		#,xmin,ymin,xmax,ymax,
		p1 = (r.xmin,r.ymin)
		p2 = (r.xmax, r.ymin)
		p3 = (r.xmin, r.ymax)
		p4 = (r.xmax,r.ymax)
		bbox_quadri= NamedQuadri([p1,p2,p3,p4], "irrelevant" )

		for q in quadrilaters:
			if bbox_quadri.intersects(q):
				df.loc[index, q.label] = True
	return df


def stats_from_video(vid_path, yolodf):
	#get stores from video
	quadrilaters = rect_painter(vid_path)


	#calc intersections with peoples
	#yolodf = calc_bbox_intersect(yolodf, quadrilaters)
	play_calc_intersects(vid_path, yolodf, zones=quadrilaters)
	#do some stats

	#heatmap
	#motion_heatmap(vid_path)

if __name__ == "__main__":
	vid = "/Users/miguelreis/Documents/work/other/okli/yolov4-deepsort/data/video/22.mp4"
	#vid = "/Users/miguelreis/Documents/work/other/okli/yolov4-deepsort/data/video/test.mp4"
	csv_out= "/Users/miguelreis/Documents/work/other/okli/yolov4-deepsort/data/out-csv/22.csv"

	df = pd.read_csv(csv_out)
	#stats_from_video(vid, df)
	motion_heatmap(vid)

	#heatmap()
	#rect_painter()