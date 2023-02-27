import torch
import cv2
import numpy as np
import argparse

class PeopleDetector:
    def __init__(self, social_distance_threshold=20, yolo_model_version = 'yolov5s'):
        self.yolo_model = torch.hub.load('ultralytics/yolov5', yolo_model_version, pretrained=True)
        self.people_info = list()
        self.social_distance_threshold = 100
        self.violation_count = 0

    def draw_text(self, frame, text, origin):
        cv2.putText(frame, 
                    text, 
                    origin, 
                    cv2.FONT_HERSHEY_SIMPLEX, #font style
                    0.7, #font scale
                    (0, 0, 255), #font color
                    1,  #font thickness
                    cv2.LINE_AA
                    )
        return frame

    def draw_foot_market(self, frame):
        for info in self.people_info:
            center_x = (info['box_vals'][1][0] + info['box_vals'][0][0]) // 2
            center_y = info['box_vals'][1][1]

            if info['is_violating']:
                marker_color = (54, 67, 244) 
            else:
                marker_color = (74, 195, 139)

            cv2.ellipse(frame, (center_x, center_y - 15), (40, 25), 0, 0, 360, marker_color, 5)
        
        return frame
    
    def draw_violation_bounding_box(self, frame):

        violation_centers = list()
        for info in self.people_info:
            local_centers = list()
            info['checked'] = True
            center_1 = info['center']
            if info['is_violating']:
                local_centers.append(center_1)
                for near_info in self.people_info:
                    center_2 = near_info['center']
                    if not near_info['checked'] and info['is_violating'] and self.calc_euclidean_dist(center_1, center_2) != 0 and self.calc_euclidean_dist(center_1, center_2) < 200:
                        local_centers.append(center_2)
                        near_info['checked'] = True
                
                if len(local_centers) > 1:
                    violation_centers.append(local_centers)

        for centers in violation_centers:
            x_center_mean = np.mean(np.array(centers)[:, 0])      
            y_center_mean = np.mean(np.array(centers)[:, 1])             
            cv2.circle(frame, (int(x_center_mean), int(y_center_mean)), 20, (0,0, 213), -1)   
            cv2.putText(frame, 
                        "Alert", 
                        (int(x_center_mean-15), int(y_center_mean+5)), 
                        cv2.FONT_HERSHEY_SIMPLEX, #font style
                        0.4, #font scale
                        (255, 255, 255), #font color
                        1,  #font thickness
                        cv2.LINE_AA
                        )
        return frame


    def draw_info(self, frame):

        frame = self.draw_foot_market(frame)
        frame = self.draw_violation_bounding_box(frame)
        frame = self.draw_text(frame, "People Count : "+ str(len(self.people_info)), (10, 20))
        frame = self.draw_text(frame, "People Violating : "+ str(self.violation_count), (10, 40))
        

        return frame

    def calc_euclidean_dist(self, point_1, point_2):
        return ((point_2[0]-point_1[0])**2 + (point_2[1]-point_1[1])**2)**0.5

    def check_social_distancing(self):
        for info in self.people_info:
            person_center = info['center']
            for next_info in self.people_info:
                near_person_center = next_info['center']
                if self.calc_euclidean_dist(person_center, near_person_center) != 0.0 and self.calc_euclidean_dist(person_center, near_person_center) < self.social_distance_threshold:
                    info['is_violating'] = True
                    self.violation_count += 1
                    break

    def process_frames(self, frame):

        # clearing previous ppl info
        self.people_info.clear()
        self.violation_count = 0

        results = self.yolo_model(frame)
        result_data = results.pandas().xyxy[0]

        xmin_vals = result_data['xmin'].values.tolist()
        ymin_vals = result_data['ymin'].values.tolist()
        xmax_vals = result_data['xmax'].values.tolist()
        ymax_vals = result_data['ymax'].values.tolist()
        object_classes = result_data['name'].values.tolist()
        confidence_vals = result_data['confidence'].values.tolist()

        for indx, object in enumerate(object_classes):
            if object == "person" and confidence_vals[indx] > 0.5:
                center_x = (int(xmax_vals[indx]) + int(xmin_vals[indx])) // 2
                center_y = ( int(ymax_vals[indx]) + int(ymin_vals[indx])) // 2

                ppl_info = {
                    'person_id' : 0,
                    'box_vals' : [(int(xmin_vals[indx]), int(ymin_vals[indx])), (int(xmax_vals[indx]), int(ymax_vals[indx]))],
                    'center' : (center_x, center_y),
                    'is_violating': False,
                    'checked': False
                }

                self.people_info.append(ppl_info)

        self.check_social_distancing()

        return self.draw_info(frame)  

def main(args):

    people_detector = PeopleDetector(args.yolo_model)
    capture = cv2.VideoCapture(args.source)
    store_output = list()

    if not capture.isOpened():
        print("Error opening the video file !")
        return

    while capture.isOpened():
        success, frame = capture.read()

        if not success:
            break

        output_frame = people_detector.process_frames(frame)

        cv2.imshow("Output", output_frame)

        if args.save_result:
            store_output.append(output_frame)       

        if cv2.waitKey(24) == ord('q'):
            break
        else:
            continue
    
    # Save results (image with detections)
    if args.save_result:
        if capture:  # video
            fps = capture.get(cv2.CAP_PROP_FPS)
            w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        else:  # stream
            fps, w, h = 30, output_frame.shape[1], output_frame.shape[0]

        save_path = "results/output_vid.mp4"
        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        for frame in store_output:
            vid_writer.write(frame)
            
        vid_writer.release()
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COVID-19 social distancing violation detector")
    parser.add_argument('--source', type=str, help="source file or 0 for webcam")
    parser.add_argument('--yolo_model', type=str, help="yolo model version (yolov5s or yolov5m or yolov5l)")
    parser.add_argument('--save_result', type=bool, help="save output")

    args = parser.parse_args()

    print(args.source)

    if args.source == None:
        args.source = "inputs/mall_area.mp4"

    main(args)